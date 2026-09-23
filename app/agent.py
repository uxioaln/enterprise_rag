# -*- coding: utf-8 -*-
"""工具按需加载 Agent（基于 LangGraph StateGraph 实现，单步，不接 checkpointer）。

代码模式参考 deer-flow 的 Skills 渐进式披露机制，用 LangGraph 的 StateGraph
替代手写 if-else 编排，使流程成为一张可被 LangGraph 运行时驱动的图。

对应关系：
  RagToolSpec            <- deerflow.skills.types.Skill（frontmatter 元数据 -> RAG 工具定义）
  ToolCatalog.search     <- deerflow.skills.catalog.SkillCatalog.search（select:/+前缀/自由文本，MAX_RESULTS=5）
  build_describe_tool    <- deerflow.skills.describe.build_describe_skill_tool（闭包，按需返回完整定义）
  <tool_index> 系统提示   <- deerflow.skills.describe.get_skill_index_prompt_section（仅注入菜单，不注入Schema）
  _loaded_schemas 清理    <- deerflow.agents.middlewares.skill_context（调用后剔除正文，仅留摘要）
  StateGraph 节点/边      <- LangGraph 图驱动，替代手写线性编排

三步走流程（图节点）：
  select_tool  (Step1 意图识别)：模型仅根据工具菜单选择工具，不加载详细参数；
  load_schema  (Step2 动态加载)：确认工具后从目录加载完整Schema（参数+返回值）并抽取调用参数；
  execute_tool (Step2 执行 + Step3 清理)：执行检索，结束后立即剔除完整Schema，仅保留结果摘要；
  generate     (生成)：上下文中已无完整Schema，仅凭检索结果生成最终答案。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TypedDict

from json_repair import repair_json
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

MAX_RESULTS = 5  # 单次搜索最多返回的候选数，与 deer-flow SkillCatalog 保持一致


# ── State：图的共享状态（单步路由，无多轮回环）──


class AgentState(TypedDict, total=False):
    """LangGraph 图状态。各节点读取/写入此结构，驱动 select→load→execute→generate 流转。"""
    question: str            # 用户原始问题
    kind: str                # 答案类型：string/number/boolean/names
    history: list | None      # 前序多轮问答记录，generate 节点拼入多轮上下文
    tool_name: str | None    # Step1 选定的工具名
    spec: RagToolSpec | None  # Step2 加载的工具定义（含完整Schema）
    args: dict | None         # Step2 抽取的调用参数
    results: list | None      # Step2 执行的检索结果
    answer_dict: dict | None  # generate 节点产出的最终答案
    tool_trace: list[str]    # 清理后保留的"调用结果摘要"（不含Schema）
    error: str | None         # 节点异常时写入，供下游感知


# ── RAG 工具定义：dataclass 保持 frozen，handler 为检索处理函数 ──


@dataclass(frozen=True)
class RagToolSpec:
    """RAG 工具定义：name/brief 相当于 Markdown Skills 的 frontmatter（name/description），
    schema_json 为完整 JSON Schema（parameters=入参，returns=返回值），仅在 Step2 动态加载。"""
    name: str
    brief: str
    schema_json: dict
    handler: object  # 检索处理函数，签名为 (processor, args) -> retrieval_results


# 使 RagToolSpec 在 AgentState 类型注解中可用（前向引用兜底）
AgentState.__annotations__["spec"] = RagToolSpec | None  # type: ignore[typeddict-item]


# ── 工具处理函数：全部复用 QuestionsProcessor 既有检索链路（MinerU 分块库，AGICTO 不受影响）──


def _tool_search_knowledge_base(processor, args: dict) -> list:
    # 知识库检索：按公司名与查询做向量检索（如启用 llm_reranking 则走混合检索）
    return processor._retrieve_for_company(args["company_name"], args["query"])


def _tool_get_financial_data(processor, args: dict) -> list:
    # 财务指标：将指标+年份拼成检索查询，并让数字密集的段落优先参与回答
    query = f"{args['year']}年 {args['metric']}" if args.get("year") else args["metric"]
    results = processor._retrieve_for_company(args["company_name"], query)
    return sorted(results, key=lambda r: any(ch.isdigit() for ch in r["text"]), reverse=True)


def _tool_compare_companies(processor, args: dict) -> list:
    # 多公司比较：逐公司检索后聚合上下文
    aggregated: list = []
    for company in args["companies"]:
        aggregated.extend(processor._retrieve_for_company(company, args["metric"]))
    if not aggregated:
        raise ValueError("compare_companies 未检索到任何上下文")
    return aggregated


# ── 工具注册表：Markdown Skills 定义格式映射为 RAG 工具（保持 JSON Schema 结构，内容为 RAG 工具）──

TOOL_REGISTRY = {
    "search_knowledge_base": RagToolSpec(
        name="search_knowledge_base",
        brief="在年报知识库中按公司名与查询做向量检索，返回最相关段落及页码",
        schema_json={
            "name": "search_knowledge_base",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": "公司全称，需与年报中一致"},
                    "query": {"type": "string", "description": "检索查询语句"},
                },
                "required": ["company_name", "query"],
            },
            "returns": {"type": "array", "items": {"type": "object",
                        "properties": {"page": {"type": "integer"}, "text": {"type": "string"}},
                        "description": "检索到的段落（含页码）"}},
        },
        handler=_tool_search_knowledge_base,
    ),
    "get_financial_data": RagToolSpec(
        name="get_financial_data",
        brief="定位公司年报中的精确财务指标数值（营收/利润/资产/负债等），数字密集段落优先",
        schema_json={
            "name": "get_financial_data",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "description": "财务指标名称，如 营业收入、总资产"},
                    "company_name": {"type": "string", "description": "公司全称"},
                    "year": {"type": "integer", "description": "会计年度，如 2022"},
                },
                "required": ["metric", "company_name"],
            },
            "returns": {"type": "array", "items": {"type": "object",
                        "properties": {"page": {"type": "integer"}, "text": {"type": "string"}},
                        "description": "含目标指标数值的年报段落"}},
        },
        handler=_tool_get_financial_data,
    ),
    "compare_companies": RagToolSpec(
        name="compare_companies",
        brief="对多家公司的同一指标分别检索并聚合上下文，用于比较类问题",
        schema_json={
            "name": "compare_companies",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "description": "待比较的指标"},
                    "companies": {"type": "array", "items": {"type": "string"},
                                  "description": "待比较的公司全称列表"},
                },
                "required": ["metric", "companies"],
            },
            "returns": {"type": "array", "items": {"type": "object",
                        "properties": {"page": {"type": "integer"}, "text": {"type": "string"}},
                        "description": "各公司相关段落聚合结果"}},
        },
        handler=_tool_compare_companies,
    ),
}


# ── 工具目录：纯搜索，镜像 deer-flow SkillCatalog.search 的三种查询语法 ──


class ToolCatalog:
    """不可变工具目录，纯搜索无变更（镜像 deer-flow SkillCatalog.search 的三种查询语法）。"""

    def __init__(self, registry: dict[str, RagToolSpec]):
        self.tools = tuple(registry.values())

    @property
    def names(self) -> frozenset[str]:
        # 全部工具名
        return frozenset(t.name for t in self.tools)

    def search(self, query: str) -> list[RagToolSpec]:
        """查询语法（与 deer-flow 一致）：
        "select:a,b" 精确按名选择；"+fin" 要求名称含 fin；"财务 指标" 自由文本匹配名称+简介。
        """
        query = query.strip()
        if not query:
            return []
        if query.startswith("select:"):
            wanted = {n.strip() for n in query[7:].split(",")}
            return [t for t in self.tools if t.name in wanted]
        if query.startswith("+"):
            required = query[1:].split(None, 1)[0].lower()
            return [t for t in self.tools if required in t.name.lower()][:MAX_RESULTS]
        scored = []
        for t in self.tools:
            searchable = f"{t.name} {t.brief}"
            if re.search(query, searchable, re.IGNORECASE):
                # 名称命中得2分，仅简介命中得1分
                scored.append((2 if re.search(query, t.name, re.IGNORECASE) else 1, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored][:MAX_RESULTS]


def build_describe_tool(catalog: ToolCatalog):
    """镜像 build_describe_skill_tool：闭包持有目录，是 Step2 动态加载的入口，
    被调用时才返回该工具的完整Schema（参数+返回值）。"""
    def describe_tool(name: str) -> RagToolSpec:
        matched = catalog.search(name if name.startswith(("select:", "+")) else f"select:{name}")
        if not matched:
            raise ValueError(f"工具目录中未匹配到: {name}")
        return matched[0]
    return describe_tool


def get_tool_index_prompt_section(catalog: ToolCatalog) -> str:
    """生成仅含工具菜单（名称+简要描述）的系统提示段，镜像 deer-flow <skill_index>。
    此阶段不注入任何工具的详细参数/返回值Schema。
    菜单行由 ToolDefinitionLayer（工具定义层）渲染，保持分层上下文结构。"""
    # 延迟导入，避免 app -> src 的加载期依赖
    from src.context_layers import ToolDefinitionLayer
    # 工具目录映射为工具定义层的输入：{工具名: {description, schema}}
    tool_layer = ToolDefinitionLayer({
        t.name: {"description": t.brief, "schema": t.schema_json} for t in catalog.tools
    })
    menu = tool_layer.render_menu()
    return f"""<tool_system>
你是一个RAG工具路由Agent。以下是可用工具菜单（仅名称与简要描述，不包含详细参数定义）。

**工具发现：**
1. 从 <tool_index> 中选择与用户任务最匹配的一个工具
2. 完整Schema（参数+返回值）将在确认后动态加载，当前无需了解
3. 只输出JSON：{{"tool": "<工具名>"}}
<tool_index>
{menu}
</tool_index>
</tool_system>"""


# ── 图节点：每个节点是一个普通函数，读取/写入 AgentState ──
# 节点内部继续调 processor.openai_processor.send_message（AGICTO 通道），
# 不依赖 BaseChatModel，与既有 MinerU/AGICTO 调用保持兼容。


def _make_nodes(processor, model: str):
    """工厂：绑定 processor 与 model，返回四个节点函数与目录/清理状态容器。

    _loaded_schemas 作为闭包容器承载"已加载完整Schema"，
    execute 节点执行后立即 pop 清理（镜像 skill_context 中间件的剔除语义）。
    """
    catalog = ToolCatalog(TOOL_REGISTRY)
    describe_tool = build_describe_tool(catalog)
    llm = processor.openai_processor
    _loaded_schemas: dict[str, str] = {}
    tool_trace: list[str] = []

    def _chat_json(system_content: str, human_content: str) -> dict:
        # qwen-plus 仅支持 json_object，采用 prompt 约束 + json_repair 解析
        text = llm.send_message(
            model=model, system_content=system_content, human_content=human_content)
        if isinstance(text, dict):
            return text
        return json.loads(repair_json(text, return_objects=True))

    def select_tool(state: AgentState) -> dict:
        """Step1 意图识别：仅根据工具菜单（名称+简要描述）选择工具，不加载详细参数。"""
        question = state["question"]
        system = get_tool_index_prompt_section(catalog)
        data = _chat_json(system, f'用户问题："{question}"')
        return {"tool_name": data.get("tool"), "tool_trace": tool_trace}

    def load_schema(state: AgentState) -> dict:
        """Step2 动态加载：确认工具后，从目录加载该工具的完整Schema（参数+返回值），并抽取调用参数。"""
        tool_name = state.get("tool_name")
        if not tool_name:
            raise ValueError("Step1 未选定工具")
        spec = describe_tool(tool_name)
        _loaded_schemas[spec.name] = json.dumps(spec.schema_json, ensure_ascii=False)
        # 基于完整Schema抽取调用参数（参数约束+返回值说明均来自Schema）
        system = (
            "你是RAG工具参数抽取器。只能基于用户问题抽取参数，不要编造。\n"
            "你的回答必须是JSON，并严格遵循如下Schema（只输出parameters对象）：\n"
            f"```\n{_loaded_schemas[spec.name]}\n```"
        )
        data = _chat_json(system, f'用户问题："{state["question"]}"')
        params = spec.schema_json["parameters"]
        # 兼容 LLM 偶发返回完整工具调用信封的情况（如 {"name": ..., "parameters": {...}}）：
        # 提示词要求"只输出parameters对象"，但部分模型（qwen-plus）有时仍嵌套一层
        # parameters，导致顶层没有任何参数键、必填参数被误判缺失而抛 ValueError（HTTP 500）；
        # 检测到嵌套 parameters 字典时解包取内层，正常平铺格式的原有解析逻辑保持不变
        if isinstance(data, dict) and isinstance(data.get("parameters"), dict):
            data = data["parameters"]
        args = {k: v for k, v in data.items() if k in params["properties"] and v is not None}
        for k, p in params["properties"].items():
            if k not in args and "default" in p:
                args[k] = p["default"]
        missing = [k for k in params.get("required", []) if k not in args]
        if missing:
            raise ValueError(f"工具 {spec.name} 缺少必填参数: {missing}")
        return {"spec": spec, "args": args}

    def execute_tool(state: AgentState) -> dict:
        """Step2 执行 + Step3 上下文清理：执行检索后立即从 _loaded_schemas 剔除完整Schema，
        仅保留"调用结果摘要"进入 tool_trace。"""
        spec = state["spec"]
        args = state["args"]
        results = spec.handler(processor, args)
        _loaded_schemas.pop(spec.name, None)  # Step3：从上下文剔除完整Schema
        pages = [r.get("page") for r in results[:10]]
        tool_trace.append(f"{spec.name} args={args} -> {len(results)} 条段落, 页码 {pages}")
        return {"results": results}

    def generate(state: AgentState) -> dict:
        """生成：上下文中已无完整Schema，仅携带工具调用结果（检索段落）生成最终答案。
        history 不为空时透传给 generate_answer_from_contexts，拼入多轮上下文。
        """
        results = state.get("results") or []
        contexts = [r["text"] for r in results]
        args = state.get("args") or {}
        answer_dict = processor.generate_answer_from_contexts(
            state["question"], contexts, state.get("kind", "string"),
            company_name=args.get("company_name"),
            retrieval_results=results,
            history=state.get("history"),
        )
        answer_dict["tool_trace"] = list(tool_trace)  # 仅摘要，不含Schema
        return {"answer_dict": answer_dict}

    return select_tool, load_schema, execute_tool, generate


def build_agent_graph(processor, model: str = "qwen-plus") -> CompiledStateGraph:
    """构建 LangGraph StateGraph：select_tool -> load_schema -> execute_tool -> generate -> END。

    单步线性流转，不接 checkpointer（暂不做状态持久化）。
    节点为普通函数，内部调用 processor 既有链路，不依赖 BaseChatModel。
    """
    select_tool, load_schema, execute_tool, generate = _make_nodes(processor, model)

    graph = StateGraph(AgentState)
    graph.add_node("select_tool", select_tool)
    graph.add_node("load_schema", load_schema)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("generate", generate)

    # 线性边：意图识别 -> 动态加载 -> 执行+清理 -> 生成 -> 结束
    graph.set_entry_point("select_tool")
    graph.add_edge("select_tool", "load_schema")
    graph.add_edge("load_schema", "execute_tool")
    graph.add_edge("execute_tool", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# ── 对外入口：保持与 pipeline.answer_with_tools 的调用契约一致 ──


class ToolAgent:
    """按需加载工具的 Agent：封装 LangGraph 图的构建与调用。

    用法（与 pipeline.answer_with_tools 保持兼容）：
        agent = ToolAgent(processor=processor, model="qwen-plus")
        answer_dict, contexts = agent.answer(question, kind)
    """

    def __init__(self, processor, model: str = "qwen-plus"):
        self.processor = processor
        self.model = model
        self.graph = build_agent_graph(processor, model=model)

    def answer(self, question: str, kind: str = "string",
               history: list[dict] | None = None) -> tuple[dict, list[str]]:
        """三步走主流程（由 LangGraph 图驱动）：意图识别 -> 动态加载并执行 -> 清理后生成答案。

        history 为前序多轮问答记录，不为空时在 generate 节点拼入多轮上下文，
        使本轮能感知上一轮交互。返回 (answer_dict, contexts)，与原签名一致。
        """
        initial_state: AgentState = {
            "question": question,
            "kind": kind,
            "history": history,
            "tool_name": None,
            "spec": None,
            "args": None,
            "results": None,
            "answer_dict": None,
            "tool_trace": [],
            "error": None,
        }
        final_state = self.graph.invoke(initial_state)
        answer_dict = final_state.get("answer_dict") or {}
        results = final_state.get("results") or []
        contexts = [r["text"] for r in results]
        return answer_dict, contexts
