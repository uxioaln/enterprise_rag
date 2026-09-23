# -*- coding: utf-8 -*-
"""
分层上下文管理（复用 context-fundamentals 技能模块的结构，Python 实现）

把原先混在一起的 System Prompt、工具定义、项目上下文、对话历史、动态附件拆分为
五个独立层模块（对齐 context engineering 五层分离架构），每层对应一类上下文组件，
可独立渲染、独立统计 token、独立按需加载：

- SystemPromptLayer         系统提示层：XML 分节结构（背景/指令/输出格式）
- ToolDefinitionLayer       工具定义层：渐进式披露（菜单常驻，Schema 按需激活，用完清理）
- ProjectContextLayer       项目上下文层：项目级稳定知识（知识库范围/公司清单/术语表）
- ConversationHistoryLayer  对话历史层：摘要注入（早期轮合并摘要，最近 N 轮保持完整）
- DynamicAttachmentsLayer   动态附件层：分块渲染 + 观测遮蔽（超长块截断并标注）

前三层内容跨请求稳定，组装进 system 位置，构成 PromptCache 可复用的稳定前缀；
对话历史层渲染为问题的前情提要；动态附件层逐请求变化，组装进 context 位置。

组装器 LayeredContextBuilder 复用 ContextBuilder 的优先级预算模式：
build() 按层位置组装出 system / context / question_prefix 三位结果，
get_usage_report() 输出按层分类的 token 使用报告。
"""

from __future__ import annotations

import json
from typing import Optional


# ---------------------------------------------------------------------------
# token 估算（复用 context_manager.estimate_token_count，按中文场景本地化）
# ---------------------------------------------------------------------------

def estimate_token_count(text: str) -> int:
    """粗略 token 估算（仅用于预算观测，不做硬性截断依据）。

    参考系数：英文 ~4 字符/token；中文/日文/韩文 ~2 字符/token
    （context-fundamentals SKILL.md Gotcha #2：非英文文本 1-2 字符/token）。
    生产系统的硬预算应改用真实 tokenizer。
    """
    if not text:
        return 0
    # 统计 CJK 字符数，其余按 ASCII 折算
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk // 2 + other // 4


# ---------------------------------------------------------------------------
# 层基类：每层独立渲染、独立统计 token
# ---------------------------------------------------------------------------

class ContextLayer:
    """上下文层基类：一个子类对应一层上下文组件。"""

    name = "context_layer"       # 层标识（决定组装位置）
    priority = 0                 # 预算紧张时的保留优先级，数值越大越优先保留

    def render(self) -> str:
        """把该层内容渲染为文本段（空层返回空字符串）。"""
        raise NotImplementedError

    def token_count(self) -> int:
        """该层的 token 估算（用于使用报告）。"""
        return estimate_token_count(self.render())


# ---------------------------------------------------------------------------
# 层 1：系统提示层（复用 context-components.md 的 XML 分节结构）
# ---------------------------------------------------------------------------

class SystemPromptLayer(ContextLayer):
    """系统提示层：按 <BACKGROUND_INFORMATION> / <INSTRUCTIONS> / <OUTPUT_DESCRIPTION>
    三段 XML 分节组织（分节边界帮助模型定位，关键约束置于首尾注意力强区）。

    兼容模式：from_raw_prompt(raw) 把现有未分节的 system prompt 整体作为
    指令段透传，render() 输出与原文完全一致，保证已调优链路行为不变。
    """

    name = "system_prompt"
    priority = 10  # 系统提示常驻会话，优先级最高

    def __init__(self, background: str = "", instructions: str = "",
                 output_format: str = "", raw: str = "") -> None:
        self.background = background        # 域背景与项目细节
        self.instructions = instructions    # 核心行为指令
        self.output_format = output_format  # 输出格式与质量标准
        self.raw = raw                      # 兼容：现有完整 system prompt（原样透传）

    @classmethod
    def from_raw_prompt(cls, raw_prompt: str) -> "SystemPromptLayer":
        """兼容层：现有完整 system prompt 原样透传（不加分节标签，行为不变）。"""
        return cls(raw=raw_prompt)

    def render(self) -> str:
        # 兼容模式：上游已调优的完整 system prompt，直接透传
        if self.raw:
            return self.raw
        # 分节模式：按提供与否拼装 XML 分节；三段全空时返回空字符串
        parts: list[str] = []
        if self.background:
            parts.append(f"<BACKGROUND_INFORMATION>\n{self.background}\n</BACKGROUND_INFORMATION>")
        if self.instructions:
            parts.append(f"<INSTRUCTIONS>\n{self.instructions}\n</INSTRUCTIONS>")
        if self.output_format:
            parts.append(f"<OUTPUT_DESCRIPTION>\n{self.output_format}\n</OUTPUT_DESCRIPTION>")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 层 2：工具定义层（复用 activate_skill_context 渐进式披露模式）
# ---------------------------------------------------------------------------

class ToolDefinitionLayer(ContextLayer):
    """工具定义层：菜单（名称+一句话描述）常驻上下文，完整 Schema 按需激活。

    三步渐进式披露（对应 Agent 三步走）：
    1. render() 默认仅输出工具菜单（低成本常驻）；
    2. activate(tool_name) 确认使用后，render() 才包含该工具的完整 Schema；
    3. clear() 调用结束后剔除全部 Schema，仅回到菜单态（上下文清理）。
    """

    name = "tool_definitions"
    priority = 8  # 工具指引次高，但 Schema 仅激活期在场

    def __init__(self, tools: dict[str, dict]) -> None:
        """tools: {工具名: {"description": 一句话描述, "schema": 完整 JSON Schema}}"""
        self.tools = tools
        self._activated: set[str] = set()  # 已激活完整 Schema 的工具名集合

    def activate(self, tool_name: str) -> None:
        """按需激活某工具的完整 Schema（渐进式披露 Step2）。"""
        if tool_name in self.tools:
            self._activated.add(tool_name)

    def deactivate(self, tool_name: str) -> None:
        """剔除单个工具的 Schema（不退回菜单态则常驻清理用）。"""
        self._activated.discard(tool_name)

    def clear(self) -> None:
        """调用结束后清理全部 Schema，仅保留菜单态（Step3 上下文清理）。"""
        self._activated.clear()

    def render_menu(self) -> str:
        """仅渲染工具菜单行（名称+描述），供 select 阶段低成本常驻。"""
        return "\n".join(f"- {name}: {spec.get('description', '')}"
                         for name, spec in self.tools.items())

    def render(self) -> str:
        # 菜单部分（常驻）
        lines = ["<TOOL_GUIDANCE>"]
        lines.append("可用工具菜单（仅名称与描述，完整定义需按需加载）：")
        lines.append(self.render_menu())
        # 已激活工具的完整 Schema（仅激活期注入，复用 skill activation pattern）
        for name in sorted(self._activated):
            schema = self.tools[name].get("schema")
            if schema is not None:
                lines.append(f"\n已激活工具 [{name}] 完整定义：")
                lines.append(json.dumps(schema, ensure_ascii=False))
        lines.append("</TOOL_GUIDANCE>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 层 3：项目上下文层（项目级稳定知识，属 PromptCache 稳定前缀侧）
# ---------------------------------------------------------------------------

class ProjectContextLayer(ContextLayer):
    """项目上下文层：承载跨请求恒定的项目级知识。

    典型内容：知识库覆盖范围描述、公司全称清单（与 subset.csv 对齐）、领域术语表。
    该层与系统提示、工具定义同属"稳定前缀"（组装进 system 位置），按项目配置
    一次构建、请求间复用，是 PromptCache 可缓存前缀的组成部分。
    """

    name = "project_context"
    priority = 9  # 项目级稳定知识，优先级仅次于系统提示

    def __init__(self, description: str = "", companies: Optional[list[str]] = None,
                 glossary: Optional[dict[str, str]] = None, raw: str = "") -> None:
        self.description = description      # 知识库覆盖范围等一句话描述
        self.companies = companies or []    # 知识库覆盖的公司全称清单
        self.glossary = glossary or {}      # 领域术语表（术语 -> 解释）
        self.raw = raw                      # 兼容：上游已拼好的项目上下文（原样透传）

    @classmethod
    def from_raw_context(cls, raw_context: str) -> "ProjectContextLayer":
        """兼容层：现有完整项目上下文原样透传（不加分节标签，行为不变）。"""
        return cls(raw=raw_context)

    def render(self) -> str:
        # 兼容模式：上游已拼好的完整项目上下文，直接透传
        if self.raw:
            return self.raw
        # 分节模式：按需拼装三段内容；全空时返回空字符串（空层不参与组装）
        parts: list[str] = []
        if self.description:
            parts.append(f"知识库范围：{self.description}")
        if self.companies:
            parts.append("覆盖公司清单：" + "、".join(self.companies))
        if self.glossary:
            parts.append("术语表：" + "；".join(f"{k}={v}" for k, v in self.glossary.items()))
        if not parts:
            return ""
        return "<PROJECT_CONTEXT>\n" + "\n".join(parts) + "\n</PROJECT_CONTEXT>"

# ---------------------------------------------------------------------------
# 层 4：对话历史层（复用 summary injection + turn representation 模式）
# ---------------------------------------------------------------------------

class ConversationHistoryLayer(ContextLayer):
    """对话历史层：前序问答渲染为"问/答"轮次序列。

    摘要注入（summary injection）：历史超过 keep_rounds 轮时，早期轮合并为
    一段纯规则摘要（不调 LLM），仅最近 keep_rounds 轮保持完整。
    渲染文本作为问题的前情提要（不进 system、不进检索上下文，职责分离）。
    """

    name = "conversation_history"
    priority = 3  # 历史仅作前情提要，优先级最低

    # 与检索上下文冲突时以检索为准的仲裁声明（与多轮上下文保持功能一致）
    _PREAMBLE = "前序对话历史（仅作为背景参考，若与本轮检索上下文冲突，以检索上下文为准）："

    def __init__(self, history: Optional[list[dict]], keep_rounds: int = 5) -> None:
        """history: [{question, final_answer}]，与 /chat 链路的会话存储格式一致。"""
        self.history = history or []
        self.keep_rounds = keep_rounds  # 保留完整记录的最近轮数

    def _summarize_rounds(self, rounds: list[dict]) -> str:
        """早期轮纯规则摘要：每轮一句，主题/结论各截取前 50 字（不加省略号）。"""
        parts = []
        for turn in rounds:
            q = (turn.get("question", "") or "")[:50]
            a = (turn.get("final_answer", turn.get("answer", "")) or "")[:50]
            if q or a:
                parts.append(f"用户之前询问了{q}，结论为{a}")
        return "；".join(parts)

    def render(self) -> str:
        if not self.history:
            return ""
        # 摘要注入：超过保留窗口时，早期轮合并为一段摘要，最近 N 轮完整保留
        if len(self.history) > self.keep_rounds:
            early = self.history[: len(self.history) - self.keep_rounds]
            recent = self.history[len(self.history) - self.keep_rounds:]
        else:
            early, recent = [], self.history
        # 渲染保留轮的"问/答"对（与原多轮拼装格式一致）
        lines: list[str] = []
        if early:
            lines.append(f"[早期对话摘要] {self._summarize_rounds(early)}")
        for turn in recent:
            q = turn.get("question", "")
            a = turn.get("final_answer", turn.get("answer", ""))
            if q or a:
                lines.append(f"问：{q}\n答：{a}")
        if not lines:
            return ""
        return self._PREAMBLE + "\n" + "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 层 5：动态附件层（复用 reference loading + observation masking 模式）
# ---------------------------------------------------------------------------

class DynamicAttachmentsLayer(ContextLayer):
    """动态附件层：承载随每次请求变化的上下文附件（检索文本块、用户上传内容、工具返回）。

    语义对齐五层模型的 Dynamic Attachments：内容逐请求变化，处于 PromptCache
    稳定前缀之外（组装进 context 位置）。渲染逻辑与原"检索结果层"完全一致：
    分块渲染（Text retrieved + 三引号包裹，块间 --- 分隔），保证既有链路输出不变。

    观测遮蔽（observation masking）：单块超过 max_block_chars 时截断并标注
    原始长度，完整内容可经外部落盘机制回读；默认 0 表示不遮蔽（保持原链路行为）。
    """

    name = "dynamic_attachments"
    priority = 5  # 动态附件按需注入，优先级中等

    def __init__(self, blocks: list[str], max_block_chars: int = 0,
                 raw: str = "") -> None:
        self.blocks = blocks                # 动态附件文本块列表（如检索返回的段落）
        self.max_block_chars = max_block_chars  # 单块遮蔽阈值（0 = 不遮蔽）
        self.raw = raw                      # 兼容：上游已拼好的完整上下文（原样透传）

    @classmethod
    def from_raw_context(cls, rag_context: str) -> "DynamicAttachmentsLayer":
        """兼容层：上游已拼好的 rag_context 原样透传（不再二次包裹，输出不变）。"""
        return cls([], raw=rag_context)

    def render(self) -> str:
        # 兼容模式：上游已按块格式拼好的完整上下文，直接透传
        if self.raw:
            return self.raw
        # 分块模式：与原链路一致的块格式（Text retrieved + 三引号包裹，块间 --- 分隔）
        parts = [f'Text retrieved: \n"""\n{self._mask_observation(b)}\n"""'
                 for b in self.blocks if b]
        return "\n\n---\n\n".join(parts)

    def _mask_observation(self, text: str) -> str:
        """复用 mask_observation 伪代码：超长块截断并替换为带标注的引用。"""
        if self.max_block_chars <= 0 or len(text) <= self.max_block_chars:
            return text
        return (text[: self.max_block_chars]
                + f"\n[观测遮蔽：原文共{len(text)}字已截断，完整内容可按文档ID回读]")


# ---------------------------------------------------------------------------
# 分层组装器（复用 context_manager.ContextBuilder 的优先级预算模式）
# ---------------------------------------------------------------------------

# 层名 -> 组装位置：system 消息 / {context} 占位符 / {question} 占位符前缀
_POSITION_BY_NAME = {
    "system_prompt": "system",
    "tool_definitions": "system",
    "project_context": "system",        # 项目级稳定知识，进 system 侧稳定前缀
    "conversation_history": "question_prefix",
    "dynamic_attachments": "context",   # 逐请求变化的动态附件，进 {context} 占位符
}


class LayeredContextBuilder:
    """分层组装器：把各层渲染结果组装为三位上下文（system / context / question_prefix）。

    build() 返回的 dict 直接对接现有 prompt 模板：
    - system            -> send_message(system_content=...)
    - context           -> user_prompt.format(context=...)
    - question_prefix   -> 拼在本轮问题前，一起进 {question} 占位符
    get_usage_report() 输出按层分类的 token 使用（复用 count_tokens_by_type 口径）。
    """

    def __init__(self, context_limit: int = 60_000) -> None:
        self.layers: dict[str, ContextLayer] = {}  # 层名 -> 层实例
        self.order: list[str] = []                  # 注册顺序（保持稳定输出）
        self.context_limit = context_limit          # token 预算上限（仅观测）

    def add_layer(self, layer: ContextLayer) -> "LayeredContextBuilder":
        """注册一个层（同名重复注册时后者覆盖，顺序保持首次位置）。"""
        if layer.name not in self.layers:
            self.order.append(layer.name)
        self.layers[layer.name] = layer
        return self

    def build(self) -> dict[str, str]:
        """按层位置组装出三位上下文文本。"""
        system_parts: list[str] = []
        context_parts: list[str] = []
        prefix_parts: list[str] = []
        for name in self.order:
            text = self.layers[name].render()
            if not text:
                continue  # 空层跳过（如无历史的对话历史层）
            position = _POSITION_BY_NAME.get(name, "system")
            if position == "system":
                system_parts.append(text)
            elif position == "context":
                context_parts.append(text)
            else:
                prefix_parts.append(text)
        return {
            "system": "\n\n".join(system_parts),
            # 多个检索层之间沿用 --- 分隔（与单层多块渲染一致）
            "context": "\n\n---\n\n".join(context_parts),
            "question_prefix": "\n\n".join(prefix_parts),
        }

    def get_usage_report(self) -> dict:
        """按层分类的 token 使用报告（用于观测与压缩触发决策）。"""
        by_layer = {name: self.layers[name].token_count() for name in self.order}
        total = sum(by_layer.values())
        return {
            "total_tokens": total,
            "limit": self.context_limit,
            "utilization": (total / self.context_limit) if self.context_limit else 0.0,
            "by_layer": by_layer,
            # 复用 ContextBuilder 的三档状态：>90% critical / >70% warning / 其余 healthy
            "status": ("critical" if total / self.context_limit > 0.9
                       else "warning" if total / self.context_limit > 0.7
                       else "healthy") if self.context_limit else "unknown",
        }


# ---------------------------------------------------------------------------
# 独立自测：五层渲染 + 组装 + 使用报告
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 层 1：系统提示（XML 分节模式）
    sys_layer = SystemPromptLayer(
        background="企业财报知识库问答",
        instructions="仅依据检索上下文作答",
        output_format="输出结构化 JSON",
    )
    # 层 2：工具定义（菜单态 -> 激活 -> 清理）
    tool_layer = ToolDefinitionLayer({
        "search_knowledge_base": {"description": "知识库向量检索",
                                 "schema": {"parameters": {"query": "string"}}},
        "get_financial_data": {"description": "财务指标查询",
                               "schema": {"parameters": {"metric": "string"}}},
    })
    # 层 3：项目上下文（知识库范围 + 公司清单 + 术语表）
    project_layer = ProjectContextLayer(
        description="中芯国际相关券商研报、财报、机构调研纪要",
        companies=["中芯国际", "华虹半导体"],
        glossary={"营收": "营业收入"},
    )
    # 层 4：对话历史（7 轮，超过 keep_rounds=5 触发摘要注入）
    history_layer = ConversationHistoryLayer(
        [{"question": f"问题{i}", "final_answer": f"答案{i}"} for i in range(1, 8)],
        keep_rounds=5,
    )
    # 层 5：动态附件（两个检索文本块）
    dynamic_layer = DynamicAttachmentsLayer(["万科2022年营收5038.4亿元。", "净负债率43.7%。"])
    # 组装 + 报告
    builder = LayeredContextBuilder()
    for layer in (sys_layer, tool_layer, project_layer, history_layer, dynamic_layer):
        builder.add_layer(layer)
    assembled = builder.build()
    print("=== 分层组装结果 ===")
    for key, value in assembled.items():
        print(f"\n[{key}]\n{value}")
    # 渐进式披露验证：激活 -> 含 Schema -> 清理 -> 仅菜单
    tool_layer.activate("get_financial_data")
    assert "已激活工具 [get_financial_data]" in tool_layer.render()
    tool_layer.clear()
    assert "已激活工具" not in tool_layer.render()
    # 摘要注入验证：早期 2 轮进摘要，最近 5 轮完整保留
    assert "[早期对话摘要]" in history_layer.render()
    assert "问：问题3" in history_layer.render() and "问：问题1" not in history_layer.render()
    # 项目上下文验证：分节渲染含公司清单与术语表，且组装进 system 位置（稳定前缀侧）
    assert "<PROJECT_CONTEXT>" in project_layer.render()
    assert "覆盖公司清单：中芯国际、华虹半导体" in assembled["system"]
    assert "术语表：营收=营业收入" in assembled["system"]
    # 项目上下文兼容模式：raw 原样透传；空层返回空字符串
    assert ProjectContextLayer.from_raw_context("已有项目说明").render() == "已有项目说明"
    assert ProjectContextLayer().render() == ""
    # 动态附件验证：分块格式与原检索链路一致，组装进 context 位置
    assert 'Text retrieved: \n"""\n万科2022年营收5038.4亿元。\n"""' in dynamic_layer.render()
    assert "万科2022年营收5038.4亿元。" in assembled["context"]
    # 动态附件兼容模式：raw 原样透传
    assert DynamicAttachmentsLayer.from_raw_context("原文上下文").render() == "原文上下文"
    # 观测遮蔽验证：超长块截断并标注原始长度
    masked = DynamicAttachmentsLayer(["x" * 300], max_block_chars=10).render()
    assert "观测遮蔽：原文共300字已截断" in masked
    # 使用报告验证：按层分类含五层键名
    report = builder.get_usage_report()
    assert {"system_prompt", "tool_definitions", "project_context",
            "conversation_history", "dynamic_attachments"} <= set(report["by_layer"])
    print("\n=== 使用报告 ===")
    print(report)
    print("\n全部自测通过")
