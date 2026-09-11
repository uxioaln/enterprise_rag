# src/query_rewriter.py
"""低置信度自动查询改写重试：查询改写器。

职责：
1. 根据评估反馈（critique）对原始查询进行改写，以提升下一轮检索质量
2. 返回 RewrittenQuery 结构化结果，包含改写后的查询、策略与理由
3. 若改写结果与当前查询相似度>0.9，强制追加限定词避免死循环

改写器复用 src.api_requests.py 的 AGICTO 调用封装（chat_completion）。
"""
from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

from json_repair import repair_json

from src.api_requests import chat_completion
import src.prompts as prompts

logger = logging.getLogger(__name__)

# 改写结果与当前查询相似度超过此阈值时，强制追加限定词避免死循环
_SIMILARITY_THRESHOLD = 0.9
# 强制追加的限定词后缀，用于打破相似度过高的死循环
_FORCED_QUALIFIER = "（请提供更详细的信息）"


@dataclass
class RewrittenQuery:
    """查询改写结果。

    rewritten_query：改写后的查询文本；
    rewrite_strategy：改写策略（expand/refine/decompose/rephrase）；
    rationale：改写理由，说明为何选择该策略。
    """
    rewritten_query: str = ""
    rewrite_strategy: str = ""
    rationale: str = ""

    def to_dict(self) -> dict:
        """转为字典，便于 JSON 序列化与存储。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RewrittenQuery":
        """从字典构造（忽略未知键），用于反序列化。"""
        return cls(
            rewritten_query=str(data.get("rewritten_query", "")),
            rewrite_strategy=str(data.get("rewrite_strategy", "")),
            rationale=str(data.get("rationale", "")),
        )


def _compute_similarity(text_a: str, text_b: str) -> float:
    """计算两段文本的相似度（基于 difflib SequenceMatcher）。

    返回 0-1 的浮点数，1 表示完全相同。
    """
    if not text_a or not text_b:
        return 0.0
    return difflib.SequenceMatcher(None, text_a, text_b).ratio()


def _build_docs_summary(docs: List[str], max_chars: int = 1000) -> str:
    """把上一轮检索到的文档片段摘要为短文本，供改写参考。"""
    if not docs:
        return "（上一轮未检索到相关文档）"
    parts: List[str] = []
    total = 0
    for i, doc in enumerate(docs[:5]):  # 仅取前5条避免过长
        snippet = doc[:200] + "..." if len(doc) > 200 else doc
        parts.append(f"[片段{i+1}] {snippet}")
        total += len(snippet)
        if total >= max_chars:
            break
    return "\n".join(parts)


def _force_append_qualifier(query: str) -> str:
    """强制追加限定词，打破相似度过高的死循环。"""
    return f"{query} {_FORCED_QUALIFIER}"


def rewrite_query(
    original_query: str,
    current_query: str,
    critique: str,
    prev_docs_summary: str = "",
    model: str = "qwen3.8-max",
) -> RewrittenQuery:
    """根据评估反馈改写查询。

    流程：
    1. 构建改写 Prompt（原始问题 + 当前问题 + 评估反思 + 上一轮文档摘要）
    2. 调用 AGICTO chat_completion 获取改写结果
    3. 解析 JSON 返回 RewrittenQuery
    4. 若改写结果与当前查询相似度>0.9，强制追加限定词
    5. 任何异常降级为在当前查询后追加限定词，不抛异常

    参数：
        original_query：用户最初的原始问题
        current_query：当前轮使用的查询（首轮与 original_query 相同）
        critique：评估器给出的反思说明
        prev_docs_summary：上一轮检索到的文档摘要
        model：改写使用的模型名，默认 qwen3.8-max

    返回：
        RewrittenQuery 结构化改写结果
    """
    try:
        user_prompt = prompts.QueryRewritePrompt.user_prompt.format(
            original_query=original_query,
            current_query=current_query,
            critique=critique,
            prev_docs_summary=prev_docs_summary or _build_docs_summary([]),
        )
        raw_response = chat_completion(
            model=model,
            system_content=prompts.QueryRewritePrompt.system_prompt,
            human_content=user_prompt,
        )
        repaired = repair_json(raw_response)
        parsed = json.loads(repaired)
        result = RewrittenQuery.from_dict(parsed)

        # 相似度检测：改写结果与当前查询过于相似时强制追加限定词
        if result.rewritten_query:
            similarity = _compute_similarity(result.rewritten_query, current_query)
            if similarity > _SIMILARITY_THRESHOLD:
                logger.info(
                    "改写结果与当前查询相似度 %.2f > %.1f，强制追加限定词",
                    similarity, _SIMILARITY_THRESHOLD,
                )
                result.rewritten_query = _force_append_qualifier(result.rewritten_query)
                result.rewrite_strategy = result.rewrite_strategy or "force_expand"
                result.rationale = f"相似度过高({similarity:.2f})，强制追加限定词。{result.rationale}"

        return result

    except Exception as exc:
        logger.warning("改写器调用失败，降级为追加限定词: %s", exc)
        return RewrittenQuery(
            rewritten_query=_force_append_qualifier(current_query),
            rewrite_strategy="fallback_force_expand",
            rationale=f"改写器调用失败({exc})，降级为追加限定词避免死循环",
        )
