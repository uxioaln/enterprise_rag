# -*- coding: utf-8 -*-
"""
两级最小上下文压缩策略
L1: 工具结果裁剪 — 将检索文档压缩为摘要，完整内容存本地文件
L2: 历史对话压缩 — 将早期对话合并为摘要，保留最近N轮完整记录

实现约束：纯规则实现，不调用任何 LLM；仅依赖 Python 标准库，可独立运行。
- L1（L1ToolResultCompressor）：检索返回的每个文档片段压缩为纯摘要
  （默认上限 800 字，约等于 300 token 中文分块的原文量）。未提供问题时
  摘要取原文前 N 字（原有逻辑）；提供问题时启用"问题感知截断"——按
  问题关键词对原文逐句打分，保留命中句及其前后上下文句，使片段中后部
  的关键数字与结论句不被盲截断丢失。完整内容落盘到本地目录，
  read_file() 可按文档ID回读（审计用途）。摘要文本不再附带文档ID与
  存储路径提示——服务端未注册 read_file 工具，附带路径只会诱导 LLM
  尝试回读失败，输出格式由调用方按 baseline 页码格式包装。
- L2（L2HistoryCompressor）：对话超过 N 轮（默认 5 轮）后，把早期轮次
  合并为一条规则化摘要，仅保留最近 N 轮的完整记录；
  保留的消息为原对象引用，不做任何内容修改。
"""

import re
from dataclasses import dataclass
from pathlib import Path


# ==================== 数据结构 ====================


@dataclass
class RetrievedDoc:
    """L1 的输入：检索返回的单个文档片段"""

    doc_id: str       # 文档唯一标识（用于落盘命名与回读）
    content: str      # 完整原文内容
    source_path: str  # 原始来源路径（知识库中的位置，作为元数据保留）


@dataclass
class Message:
    """L2 的基本单元：一条对话消息"""

    role: str     # 消息角色：system / user / assistant / tool
    content: str  # 消息文本内容


# ==================== L1: 工具结果裁剪 ====================


class L1ToolResultCompressor:
    """L1 工具结果裁剪器。

    将检索返回的文档片段压缩为纯摘要文本，完整原文写入本地存储目录
    （默认 /workspace/retrieved_docs），read_file() 可按文档ID回读（审计用途）。

    摘要生成策略（问题感知截断）：
    - 未提供问题时：退化为原有逻辑，取原文前 summary_chars 字；
    - 提供问题时：按句切分原文并逐句打分（命中问题关键词每词 +1 分、
      命中年份每年 +2 分），取得分最高的 top_k 句并连同前后各
      context_window 句上下文按原文顺序拼接，总长度超过
      summary_chars 时硬截断兜底；无任何命中时同样退化为前 N 字。
      关键数字与结论常位于片段中后部，该策略可在有限 token 内
      保留回答问题所需的句子，避免"前 200 字盲截断"丢数据。

    输出格式说明（A1 修复）：compress 返回纯摘要，不再附带
    "文档ID + 存储路径"前缀——服务端 TOOL_REGISTRY 未注册 read_file
    工具，附带路径提示会诱导 LLM 尝试回读失败并在答案中出现
    "内容已存档"等模糊表述；调用方（questions_processing）按 baseline
    的页码三引号格式包装摘要，保证 L1 开启前后 LLM 看到的上下文
    结构一致，仅正文由全文换为压缩摘要。
    """

    # 术语分隔词：问题中的实体后缀与疑问/指令动词，不作为匹配关键词
    _TERM_SEPARATORS = ("公司", "股份", "有限", "什么", "哪些", "多少", "是否",
                        "如何", "怎样", "请问", "以及", "还有", "告诉", "介绍",
                        "说明", "提到", "对比", "比较", "分析", "分别", "去年", "今年")
    # 单字虚词：出现在问题中时用于打断/滤除无区分度的字（年份由此解耦，
    # 如 "2025年境外营收" 拆为年份 2025 与关键词 "境外营收"）
    _FUNC_CHARS = set("的了是在和与及对为有之或被把将按从到这那其该各每呢吗吧地得年")

    def __init__(self, store_dir: str | Path = "/workspace/retrieved_docs",
                 summary_chars: int = 800, top_k: int = 6,
                 context_window: int = 1) -> None:
        self.store_dir = Path(store_dir)     # 完整内容的落盘目录
        self.summary_chars = summary_chars   # 摘要保留的字符数上限（默认 800 字：约等于 300 token 的中文分块原文量，压缩比约 2:1 且不丢关键数字）
        self.top_k = top_k                   # 问题感知截断：取得分最高的前 K 句（默认 6，覆盖多数关键数字句与结论句）
        self.context_window = context_window  # 问题感知截断：命中句前后各保留的上下文句数
        # 确保存储目录存在（不存在则逐级创建）
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def compress(self, docs: list[RetrievedDoc], question: str = "") -> list[str]:
        """压缩一组检索文档，返回裁剪后的纯摘要文本列表（与 docs 顺序一致）。

        question 非空时启用问题感知截断（按问题关键词选句），为空时保持
        原有"前 summary_chars 字"逻辑。完整原文同步写入
        {store_dir}/{doc_id}.txt，供审计与 read_file(doc_id) 回读；
        返回值为纯摘要——不带"文档ID/存储路径"前缀（A1 修复：服务端
        未注册 read_file 工具，附带路径提示会诱导 LLM 尝试回读失败），
        输出格式由调用方按 baseline 页码三引号格式包装。
        """
        compressed: list[str] = []
        for doc in docs:
            # 1) 完整原文写入本地文件，供审计与 read_file(doc_id) 回读
            file_path = self.store_dir / f"{doc.doc_id}.txt"
            file_path.write_text(doc.content, encoding="utf-8")
            # 2) 摘要生成：question 非空 -> 问题感知选句；为空 -> 前 N 字（原有逻辑）
            if question.strip():
                summary = self._select_relevant(doc.content, question)
            else:
                summary = doc.content[: self.summary_chars]
            # 3) 返回纯摘要（A1 修复：去掉文档ID/存储路径前缀，避免伪闭环提示干扰 LLM）
            compressed.append(summary)
        return compressed

    def _split_sentences(self, content: str) -> list[str]:
        """按中文句末标点与换行切分原文为句子列表（保留标点，滤除空白片段）。"""
        # 正则后行断言切分：句号/问号/感叹号/分号/换行留在前一句末尾
        parts = re.split(r"(?<=[。！？；\n])", content)
        return [p.strip() for p in parts if p and p.strip()]

    def _extract_keywords(self, question: str) -> tuple[list[str], list[str], list[str]]:
        """从问题中提取（关键词列表, 年份列表, 实体名列表）。

        年份为 4 位数字，作为强信号（命中句子每年 +2 分）；关键词为滤除
        实体后缀/疑问指令词/单字虚词后、长度>=2 的连续中英文数字片段
        （纯数字片段不作为关键词，交由年份逻辑处理）。
        实体名为问题中的公司/机构名（含"公司""股份""时代""银行"等后缀
        的连续中文片段），用于强制保留含实体名的句子——L1 压缩曾因丢弃
        含公司名的标题句导致 LLM 误判"公司名不在上下文中"（id=14 案例）。
        """
        # 1) 年份直接用正则提取（去重保序）
        years = list(dict.fromkeys(re.findall(r"\d{4}", question)))
        # 2) 先替换多字分隔词为空格，再逐字符滤除单字虚词
        normalized = question
        for word in self._TERM_SEPARATORS:
            normalized = normalized.replace(word, " ")
        normalized = "".join(" " if ch in self._FUNC_CHARS else ch for ch in normalized)
        # 3) 提取连续片段，过滤纯数字与单字符，去重保序
        keywords: list[str] = []
        for chunk in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", normalized):
            if len(chunk) >= 2 and not chunk.isdigit() and chunk not in keywords:
                keywords.append(chunk)
        # 4) 实体名提取：从原始问题中匹配含公司后缀的连续中文片段
        # 用于强制保留含实体名的句子（即使该句关键词得分不高）
        entity_patterns = [
            r"[\u4e00-\u9fa5]{2,}(?:股份有限公司|股份有限公司|集团|时代|银行|国际|汽车|能源|茅台)",
            r"[\u4e00-\u9fa5]{2,}(?:公司|股份)",
        ]
        entities: list[str] = []
        for pat in entity_patterns:
            for m in re.findall(pat, question):
                if m not in entities:
                    entities.append(m)
        # 去重：实体名如果与关键词重复则从关键词中移除（避免双重计分）
        for ent in entities:
            if ent in keywords:
                keywords.remove(ent)
        return keywords, years, entities

    def _keyword_hit(self, keyword: str, sentence: str) -> bool:
        """判断关键词是否命中句子。

        直接子串命中；或长关键词（>=4 字）的相邻二元词组有半数以上出现
        在句中也视为命中（容忍词尾差异，如"同比增长率"命中"同比增长8.3%"）。
        """
        if keyword in sentence:
            return True
        if len(keyword) >= 4:
            grams = [keyword[i:i + 2] for i in range(len(keyword) - 1)]
            hits = sum(1 for g in grams if g in sentence)
            if hits * 2 >= len(grams):
                return True
        return False

    def _select_relevant(self, content: str, question: str) -> str:
        """问题感知截断：选取命中问题关键词的句子及上下文窗口拼接为摘要。

        评分规则（在原有关键词+年份基础上增加两条）：
        - 命中关键词每词 +1 分（原有）
        - 命中年份每年 +2 分（原有）
        - 含实体名（公司名）的句子强制入选（新增：防止 L1 丢弃公司名
          标题句导致 LLM 误判"公司名不在上下文中"，id=14 案例）
        - 含 4 位以上数字或金融单位（亿元/万元/%/元）的句子 +1 分（新增：
          金融答案依赖精确数字，数字句丢失会导致 LLM 从参数知识补数字
          产生幻觉，id=9 案例）
        """
        # 1) 切句并提取问题关键词、年份与实体名
        sentences = self._split_sentences(content)
        keywords, years, entities = self._extract_keywords(question)
        # 2) 关键词、年份与实体名均为空（问题全是虚词）时，退化为前 N 字
        if (not keywords and not years and not entities) or not sentences:
            return content[: self.summary_chars]
        # 3) 逐句打分：关键词 +1、年份 +2、数字句 +1
        scored: list[tuple[int, int]] = []
        # 实体名命中的句子索引集合（强制入选，不受 top_k 限制）
        entity_hit_idx: set[int] = set()
        for idx, sentence in enumerate(sentences):
            score = sum(1 for k in keywords if self._keyword_hit(k, sentence))
            score += 2 * sum(1 for y in years if y in sentence)
            # 数字句加分：含 4 位以上数字或金融单位词的句子 +1
            if re.search(r"\d{4,}|\d+[,.]?\d*%|\d+[亿万亿]元?", sentence):
                score += 1
            # 实体名命中：强制标记入选
            if any(ent in sentence for ent in entities):
                entity_hit_idx.add(idx)
                score = max(score, 1)  # 确保至少 1 分以进入 scored 列表
            if score > 0:
                scored.append((idx, score))
        # 4) 无任何命中：同样退化为前 N 字
        if not scored:
            return content[: self.summary_chars]
        # 5) 取得分最高的 top_k 句（同分按原文顺序优先），并扩展前后上下文窗口
        top = sorted(scored, key=lambda t: (-t[1], t[0]))[: self.top_k]
        selected: set[int] = set()
        for idx, _ in top:
            lo = max(0, idx - self.context_window)
            hi = min(len(sentences), idx + self.context_window + 1)
            selected.update(range(lo, hi))
        # 6) 实体名命中句强制入选（即使不在 top_k 中也纳入）
        selected.update(entity_hit_idx)
        for idx in entity_hit_idx:
            lo = max(0, idx - self.context_window)
            hi = min(len(sentences), idx + self.context_window + 1)
            selected.update(range(lo, hi))
        # 7) 按原文顺序拼接，超过 summary_chars 时硬截断兜底
        summary = "".join(sentences[i] for i in sorted(selected))
        return summary[: self.summary_chars]

    def read_file(self, doc_id: str) -> str:
        """Agent 需要完整内容时调用：按文档ID读取本地存储的原文。"""
        path = self.store_dir / f"{doc_id}.txt"
        # match-case 检查文件存在性，不存在时给出清晰错误信息
        match path.is_file():
            case True:
                return path.read_text(encoding="utf-8")
            case False:
                raise FileNotFoundError(f"文档 {doc_id} 的完整内容不存在: {path}")


# ==================== L2: 历史对话压缩 ====================


class L2HistoryCompressor:
    """L2 历史对话压缩器。

    对话超过 max_rounds 轮后，把早期轮次合并为一条规则化摘要：
    "用户之前询问了[主题]，已获取[文档名称]中的相关内容，结论为[核心结论]"；
    仅保留最近 max_rounds 轮的完整记录（原消息对象原样引用，不修改内容）。
    """

    # 早期消息中没有任何 user 角色时使用的兜底摘要文案
    _FALLBACK_SUMMARY = "[对话历史摘要] 早期对话包含系统消息和工具调用结果，核心信息已整合。"

    def __init__(self, max_rounds: int = 5, excerpt_chars: int = 50) -> None:
        self.max_rounds = max_rounds        # 保留的最近对话轮数（默认 5 轮）
        self.excerpt_chars = excerpt_chars  # 摘要中主题/结论的截取长度（默认 50 字）

    def compress(self, messages: list[Message]) -> list[Message]:
        """压缩对话历史：早期轮次 -> 1 条摘要，最近 max_rounds 轮原样保留。

        未超过保留轮数时直接原样返回，不做任何压缩。
        """
        # 1) 按轮切分消息序列
        rounds = self._split_rounds(messages)
        # 2) 总轮数未超过保留窗口：不压缩，原样返回
        if len(rounds) <= self.max_rounds:
            return messages
        # 3) 超过保留窗口：早期轮次生成摘要，最近 max_rounds 轮展开为保留消息
        old_rounds = rounds[: len(rounds) - self.max_rounds]
        recent_messages = [m for r in rounds[len(rounds) - self.max_rounds:] for m in r]
        # 4) 早期轮次合并为一条摘要（挂在 assistant 角色上，作为对话记忆）
        summary = self._build_summary(old_rounds)
        # 5) 返回：1 条摘要 + 保留的最近消息（原对象引用，内容未做任何修改）
        return [Message(role="assistant", content=summary), *recent_messages]

    def _split_rounds(self, messages: list[Message]) -> list[list[Message]]:
        """将消息序列切分为轮。

        情况1（存在 user 消息）：每条 user 消息开启新一轮，其后到下一条 user
        之前的 assistant/tool 消息归属该轮；列表开头第一条 user 之前的
        system/tool 前缀消息单独成轮。
        情况2（无任何 user 消息，纯 system/tool）：按每 2 条一组切为虚拟轮，
        保证该场景仍可触发压缩与兜底摘要。
        """
        # 情况1：以 user 消息为轮起点切分
        if any(m.role == "user" for m in messages):
            rounds: list[list[Message]] = []
            current: list[Message] = []
            for msg in messages:
                if msg.role == "user":
                    # 遇到新 user：把已累积的当前轮封存，user 开启新一轮
                    if current:
                        rounds.append(current)
                    current = [msg]
                else:
                    # user 之后的 assistant/tool 消息并入当前轮
                    current.append(msg)
            if current:
                rounds.append(current)
            return rounds
        # 情况2：无 user 消息时按每 2 条一组切为虚拟轮
        return [messages[i: i + 2] for i in range(0, len(messages), 2)]

    def _build_summary(self, old_rounds: list[list[Message]]) -> str:
        """把早期轮次合并为一段规则化摘要（不调用 LLM）。"""
        # 展平早期所有消息，用于判断是否存在 user 角色
        old_messages = [m for r in old_rounds for m in r]
        # 早期消息中没有 user 角色：使用统一兜底文案
        if not any(m.role == "user" for m in old_messages):
            return self._FALLBACK_SUMMARY
        # 存在 user 角色：逐轮生成标准句式后合并为一段摘要
        parts: list[str] = []
        for round_msgs in old_rounds:
            # 该轮的 user 消息（主题来源）；无 user 的轮次（如 system 前缀轮）跳过
            user = next((m for m in round_msgs if m.role == "user"), None)
            if user is None:
                continue
            # 该轮最后一条 assistant 消息（结论来源）
            assistant = next((m for m in reversed(round_msgs) if m.role == "assistant"), None)
            # 主题：截取问题前 50 字，不足 50 字保留全部，末尾不加省略号
            topic = user.content[: self.excerpt_chars]
            # 结论：assistant 回答前 50 字（与主题同一截取规则）；无回答则标记无结论
            conclusion = (assistant.content[: self.excerpt_chars]
                          if assistant else "无明确结论")
            # 文档名称：纯规则提取（L1 标记 / 书名号 / 文件名），提取不到用"相关文档"
            doc_name = self._extract_doc_name(round_msgs)
            # 按指定句式拼接该轮摘要
            parts.append(
                f"用户之前询问了{topic}，已获取{doc_name}中的相关内容，结论为{conclusion}"
            )
        # 多轮摘要按行合并为一段
        return "\n".join(parts)

    def _extract_doc_name(self, round_msgs: list[Message]) -> str:
        """纯规则提取该轮对话涉及的文档名称（不调用 LLM）。

        提取优先级：
        1. L1 压缩产物中的文档ID标记，如 "[文档ID: doc_001]"；
        2. 书名号《...》中的文档名；
        3. 常见文件名（含 .pdf/.txt/.md/.docx 扩展名）；
        4. 均未命中时返回"相关文档"。
        """
        # 合并该轮所有消息文本，便于统一做模式匹配
        text = "\n".join(m.content for m in round_msgs)
        # 按优先级依次尝试三种模式
        for pattern in (r"\[文档ID:\s*([^\]]+)\]",
                        r"《([^》]+)》",
                        r"[A-Za-z0-9_\-.·\u4e00-\u9fff]+\.(?:pdf|txt|md|docx)"):
            match = re.search(pattern, text)
            if match:
                # 带捕获组的返回组内容（文档ID/书名），无捕获组的返回整段匹配（文件名）
                return match.group(1) if match.groups() else match.group(0)
        # 兜底：未提取到任何文档标识
        return "相关文档"


# ==================== 测试用例 ====================


if __name__ == "__main__":
    # ------------------------------------------------------------
    # L1 测试：构造 3 条 RetrievedDoc，调用压缩，验证文件是否生成
    # ------------------------------------------------------------
    print("=" * 70)
    print("L1 测试：工具结果裁剪")
    print("=" * 70)
    # 默认存储目录为 /workspace/retrieved_docs；本地运行环境可能没有该路径的
    # 写权限，测试改用当前目录下的 retrieved_docs_test 验证同一逻辑
    test_store_dir = Path("retrieved_docs_test")
    # 清理旧测试产物，避免历史文件干扰"是否生成"的验证
    if test_store_dir.exists():
        for old_file in test_store_dir.glob("*.txt"):
            old_file.unlink()
    # 构造 L1 压缩器（摘要保留前 200 字）
    l1 = L1ToolResultCompressor(store_dir=test_store_dir, summary_chars=200)

    # 构造 3 条 RetrievedDoc（内容均超过 200 字，用于验证摘要截断）
    docs = [
        RetrievedDoc(
            doc_id="doc_001",
            source_path="data/stock_data/pdf_reports/万科2022年年度报告.pdf",
            content=("万科企业股份有限公司2022年年度报告披露：报告期内公司实现营业收入5038.4亿元，"
                     "同比增长11.27%；归属于上市公司股东的净利润226.2亿元。房地产开发业务实现"
                     "结算收入3468.7亿元，占营业收入的比例为68.8%；物业服务体系之一的万物云实现"
                     "营业收入301.1亿元，同比增长27.0%。经营性现金流净额27.5亿元，连续14年为正。"
                     "截至报告期末，公司净负债率为43.7%，持有货币资金1372.1亿元，覆盖一年内到期的"
                     "有息负债的倍数为2.4倍。年内公司新增加开发项目41个，总规划计容建筑面积约"
                     "473.5万平方米。"),
        ),
        RetrievedDoc(
            doc_id="doc_002",
            source_path="data/stock_data/pdf_reports/中芯国际2022年业绩报告.pdf",
            content=("中芯国际2022年年度业绩报告：全年实现营业收入495.16亿元，同比增长39.3%，"
                     "创历史新高；毛利率为38.0%，同比提升9.0个百分点；归属于上市公司股东的净利润"
                     "121.33亿元，同比增长13.0%。按技术节点划分，来自90纳米及以下制程的晶圆代工"
                     "业务营收占比为62.2%，其中55/65纳米贡献22.8%、40/45纳米贡献15.4%、FinFET及"
                     "28纳米以下合计贡献15.8%。全年晶圆付运量（折合8英寸约当晶圆）达到709.7万片，"
                     "产能利用率保持较高水平。公司资本开支约为446.6亿元，主要投向上海临港、北京、"
                     "深圳及天津的12英寸晶圆厂建设项目。"),
        ),
        RetrievedDoc(
            doc_id="doc_003",
            source_path="data/stock_data/pdf_reports/机构调研纪要.pdf",
            content=("机构调研纪要：公司管理层在业绩说明会上表示，2023年将继续坚持稳健经营策略，"
                     "房地产开发业务聚焦核心城市与优质地段，严控拿地成本与杠杆水平。关于市场关注度"
                     "较高的债务问题，管理层回应称公司已提前完成2023年到期境内公开市场债券的置换与"
                     "偿还安排，融资渠道保持畅通。经营性物业方面，公司计划未来三年将购物中心与"
                     "管理型住宅社区的运营规模进一步提升，目标成为行业领先的城乡建设与生活服务"
                     "提供商。同时公司将继续探索经营性业务分拆上市的可能性。"),
        ),
    ]

    # 调用压缩，得到裁剪后的摘要文本列表
    compressed = l1.compress(docs)
    print(f"\n压缩结果（共 {len(compressed)} 条）：")
    for text in compressed:
        print("-" * 70)
        print(text)

    # 验证 1：每条压缩文本均为纯摘要（A1 修复：无文档ID前缀、无存储路径提示，
    # 无问题时退化为前 N 字：输出恰为原文前 summary_chars 字）
    for doc, text in zip(docs, compressed):
        assert "[文档ID:" not in text, f"{doc.doc_id} 不应包含文档ID前缀"
        assert "完整内容已存储" not in text, f"{doc.doc_id} 不应包含存储路径提示"
        assert text == doc.content[:200], f"{doc.doc_id} 纯摘要应等于原文前200字"
    # 验证 2：完整内容文件已生成，且内容与原文一致
    for doc in docs:
        file_path = test_store_dir / f"{doc.doc_id}.txt"
        assert file_path.is_file(), f"文件未生成: {file_path}"
        assert file_path.read_text(encoding="utf-8") == doc.content, f"{doc.doc_id} 落盘内容与原文不一致"
    # 验证 3：read_file 能按文档ID读回完整原文
    assert l1.read_file("doc_002") == docs[1].content, "read_file 回读内容与原文不一致"
    print("-" * 70)
    print(f"文件生成验证：{test_store_dir.resolve()} 下已生成 "
          f"{len(list(test_store_dir.glob('*.txt')))} 个完整内容文件")

    # ------------------------------------------------------------
    # L1 问题感知截断测试：关键数字句位于原文前 200 字之后，
    # 验证提供问题后摘要能命中关键句，而非盲取前 N 字
    # ------------------------------------------------------------
    print("=" * 70)
    print("L1 测试：问题感知截断")
    print("=" * 70)
    # 构造文档：前 206 字为与问题无关的经营概述（两段各 103 字），
    # 关键比例句（含 38.65% / 28.55%）在其之后
    noise = ("公司报告期内整体经营情况平稳，主营业务保持稳定发展，"
             "管理层持续优化治理结构与内部控制体系，积极履行社会责任，"
             "推动绿色低碳运营与数字化转型，各项基础管理工作有序开展，"
             "组织能力与人才队伍建设稳步推进，品牌影响力持续提升。") * 2
    key_sentence = "2025年境外营收占总营收比例为38.65%，而2024年为28.55%。"
    tail_sentence = "公司境外业务覆盖欧洲、东南亚等多个市场。"
    qa_doc = RetrievedDoc(
        doc_id="doc_qa",
        source_path="data/stock_data/pdf_reports/测试年报.pdf",
        content=noise + key_sentence + tail_sentence,
    )
    # 测试前提：关键句必须位于前 200 字之外，否则用例失效
    assert "38.65%" not in qa_doc.content[:200], "测试前提失效：关键句应位于前200字之外"
    qa_question = "比亚迪2025年境外营收占总营收的比例是多少？"
    # 1) 不提供问题：退化为前 200 字，关键数字句不在摘要中（原有盲截断行为）
    blind = l1.compress([qa_doc])[0]
    assert "38.65%" not in blind, "盲截断摘要不应包含第200字之后的关键句"
    # 2) 提供问题：应命中关键数字句（含 38.65% 与 28.55% 两个年份比例）
    aware = l1.compress([qa_doc], question=qa_question)[0]
    assert "38.65%" in aware, "问题感知摘要缺少关键数字 38.65%"
    assert "28.55%" in aware, "问题感知摘要缺少对比数字 28.55%"
    # 3) 与问题无关的提问（无任何关键词/年份命中）：退化为前 N 字
    unrelated = l1.compress([qa_doc], question="谢谢")[0]
    assert "38.65%" not in unrelated, "无关问题应退化为前N字摘要"
    print(f"盲截断摘要（前200字，未命中关键句，长度 {len(blind)}）")
    print(f"问题感知摘要（命中关键句）：{aware[:150]}")
    print("L1 问题感知截断测试通过")
    print("L1 测试通过\n")

    # ------------------------------------------------------------
    # L2 测试：构造 14 条 Message（7轮对话），max_rounds=5，验证返回 11 条
    # ------------------------------------------------------------
    print("=" * 70)
    print("L2 测试：历史对话压缩（14条消息=7轮，max_rounds=5）")
    print("=" * 70)
    # 构造 L2 压缩器（保留最近 5 轮）
    l2 = L2HistoryCompressor(max_rounds=5)

    # 构造 7 轮对话（user/assistant 交替，共 14 条消息）
    qa_pairs = [
        # 第1轮：问题超过 50 字，验证主题截取；回答含书名号文档名
        ("请详细介绍万科企业股份有限公司2022年年度报告中披露的营业收入、归属于母公司股东的净利润以及各主要业务板块的开发经营情况",
         "根据《万科2022年年度报告》，公司2022年实现营业收入5038.4亿元，同比增长11.27%；归属于上市公司股东的净利润226.2亿元。"),
        # 第2轮：问题不足 50 字，验证保留全部；回答含 L1 压缩产物标记
        ("中芯国际的晶圆代工业务在2022年的表现如何？",
         "根据检索结果 [文档ID: doc_002] 的内容，中芯国际2022年晶圆代工业务收入创历史新高，90纳米及以下制程营收占比达62.2%。"),
        ("万科的净负债率是多少？",
         "截至2022年末，万科净负债率为43.7%，货币资金1372.1亿元，覆盖短债倍数2.4倍。"),
        ("中芯国际的资本开支投向了哪些项目？",
         "2022年资本开支约446.6亿元，主要投向上海临港、北京、深圳及天津的12英寸晶圆厂建设。"),
        ("管理层对2023年经营有什么规划？",
         "管理层表示将继续坚持稳健经营，聚焦核心城市优质地段，严控拿地成本与杠杆水平。"),
        ("万科的物业业务收入是多少？",
         "万物云2022年实现营业收入301.1亿元，同比增长27.0%。"),
        ("中芯国际的毛利率是多少？",
         "2022年毛利率为38.0%，同比提升9.0个百分点。"),
    ]
    messages = [Message(role=role, content=text)
               for q, a in qa_pairs
               for role, text in (("user", q), ("assistant", a))]

    # 调用压缩
    result = l2.compress(messages)
    print(f"\n压缩前消息数: {len(messages)}，压缩后消息数: {len(result)}")
    print("-" * 70)
    print("摘要消息内容：")
    print(result[0].content)
    print("-" * 70)

    # 验证 1：返回长度为 11（1条摘要 + 10条最近消息）
    assert len(result) == 11, f"期望返回 11 条，实际返回 {len(result)} 条"
    # 验证 2：第一条为规则化摘要消息
    assert result[0].role == "assistant"
    assert "用户之前询问了" in result[0].content
    # 验证 3：最近 10 条消息与原始消息逐条一致（内容未被修改）
    assert ([(m.role, m.content) for m in result[1:]]
            == [(m.role, m.content) for m in messages[4:]]), "保留消息内容被修改"
    # 验证 4：摘要覆盖了被压缩的前 2 轮主题（第1轮截取前50字、第2轮保留全部）
    assert "请详细介绍万科企业股份有限公司2022年年度报告中披露的营业收入、归属于母公司股东的净利润以及各主" in result[0].content
    assert "中芯国际的晶圆代工业务在2022年的表现如何？" in result[0].content
    print(f"返回结构验证：1 条摘要 + {len(result) - 1} 条最近消息 = {len(result)} 条")
    print("L2 测试通过\n")

    # ------------------------------------------------------------
    # L2 补充验证：早期消息无 user 角色时使用兜底摘要文案
    # ------------------------------------------------------------
    print("=" * 70)
    print("L2 补充验证：早期消息无 user 角色（兜底文案）")
    print("=" * 70)
    # 构造 12 条纯 system/tool 消息（无 user），按每 2 条一组切为 6 个虚拟轮
    fallback_messages = [Message(role=role, content=content)
                         for role, content in [
        ("system", "你是企业知识库问答助手"), ("tool", "检索完成，返回10个片段"),
        ("system", "当前会话上下文清理策略已启用"), ("tool", "工具调用结果已裁剪为摘要"),
        ("system", "知识库索引加载完毕"), ("tool", "BM25检索返回8个候选段落"),
        ("system", "检索上下文已注入"), ("tool", "页码校验通过，共12处引用"),
        ("system", "进入答案生成阶段"), ("tool", "结构化答案生成完毕"),
        ("system", "会话状态已同步"), ("tool", "历史记录已写入存储"),
    ]]
    fallback_result = l2.compress(fallback_messages)
    print(f"\n压缩前消息数: {len(fallback_messages)}，压缩后消息数: {len(fallback_result)}")
    print("摘要消息内容：")
    print(fallback_result[0].content)
    # 验证：兜底文案 + 最近 10 条原始消息
    assert fallback_result[0].content == L2HistoryCompressor._FALLBACK_SUMMARY
    assert len(fallback_result) == 11
    assert [(m.role, m.content) for m in fallback_result[1:]] == \
           [(m.role, m.content) for m in fallback_messages[2:]]
    print("L2 兜底文案验证通过")
