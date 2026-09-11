# src/retrieval_evaluator.py
"""低置信度自动查询改写重试：检索质量与答案置信度评估器。

职责：
1. 评估给定 (query, docs, answer) 三元组的整体质量
2. 返回 ConfidenceScore 结构化评分，供重试循环决策
3. 失败时降级为基于重排分数的启发式评分，不抛异常阻塞主流程

评估器复用 src.api_requests.py 的 AGICTO 调用封装（chat_completion）。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

from json_repair import repair_json

from src.api_requests import chat_completion
import src.prompts as prompts

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """置信度评估结果。

    overall：综合置信度（0-1），retrieval/faithfulness/completeness 的加权综合；
    should_retry：overall 低于阈值时为 True，表示需要触发查询改写重试。
    """
    overall: float = 0.0
    retrieval_confidence: float = 0.0
    faithfulness: float = 0.0
    completeness: float = 0.0
    critique: str = ""
    should_retry: bool = False

    def to_dict(self) -> dict:
        """转为字典，便于 JSON 序列化与存储。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ConfidenceScore":
        """从字典构造（忽略未知键），用于反序列化。"""
        return cls(
            overall=float(data.get("overall", 0.0)),
            retrieval_confidence=float(data.get("retrieval_confidence", 0.0)),
            faithfulness=float(data.get("faithfulness", 0.0)),
            completeness=float(data.get("completeness", 0.0)),
            critique=str(data.get("critique", "")),
            should_retry=bool(data.get("should_retry", False)),
        )


def _build_context_text(docs: List[str], max_chars: int = 4000) -> str:
    """把检索到的文档片段拼接为紧凑的上下文文本，截断防止超长。"""
    parts: List[str] = []
    total = 0
    for doc in docs:
        snippet = doc[:max_chars] if len(doc) > max_chars else doc
        parts.append(snippet)
        total += len(snippet)
        if total >= max_chars:
            break
    return "\n---\n".join(parts)


def _build_answer_text(answer: dict) -> str:
    """从结构化 answer_dict 提取答案摘要文本，供评估使用。"""
    final = answer.get("final_answer", "")
    summary = answer.get("reasoning_summary", "")
    parts = []
    if summary:
        parts.append(f"[推理摘要] {summary}")
    if final:
        parts.append(f"[最终答案] {final}")
    return "\n".join(parts) if parts else str(answer)


def _heuristic_score(query: str, docs: List[str], answer: dict) -> ConfidenceScore:
    """启发式降级评分：基于答案是否为空、文档是否为空等简单规则。

    当 AGICTO 评估调用失败时使用，不抛异常，返回保守评分。
    """
    answer_text = str(answer.get("final_answer", "")).strip()
    has_docs = len(docs) > 0
    has_answer = len(answer_text) > 0 and answer_text.upper() != "N/A"

    if not has_docs or not has_answer:
        # 无文档或无答案：低置信度，应重试
        return ConfidenceScore(
            overall=0.2,
            retrieval_confidence=0.1 if not has_docs else 0.3,
            faithfulness=0.1,
            completeness=0.1,
            critique="启发式评分：文档或答案缺失，建议重试" if (not has_docs or not has_answer) else "启发式评分",
            should_retry=True,
        )

    # 有文档且有答案：中等置信度，不强制重试
    return ConfidenceScore(
        overall=0.75,
        retrieval_confidence=0.7,
        faithfulness=0.75,
        completeness=0.7,
        critique="启发式评分：文档与答案均存在，降级评估无法精确判断质量",
        should_retry=False,
    )


def evaluate(
    query: str,
    docs: List[str],
    answer: dict,
    model: str = "qwen3.8-max",
) -> ConfidenceScore:
    """评估 (query, docs, answer) 三元组的整体质量。

    流程：
    1. 构建评估 Prompt（问题 + 上下文 + 答案）
    2. 调用 AGICTO chat_completion 获取结构化评分
    3. 解析 JSON 返回 ConfidenceScore
    4. 任何异常降级为启发式评分，不抛异常阻塞主流程

    参数：
        query：用户原始问题
        docs：检索到的文档片段列表
        answer：Pipeline 生成的结构化答案 dict（含 final_answer / reasoning_summary 等）
        model：评估使用的模型名，默认 qwen3.8-max

    返回：
        ConfidenceScore 结构化评分
    """
    if not docs or not answer:
        return _heuristic_score(query, docs, answer)

    context_text = _build_context_text(docs)
    answer_text = _build_answer_text(answer)

    try:
        user_prompt = prompts.RetrievalEvaluationPrompt.user_prompt.format(
            question=query,
            context=context_text,
            answer=answer_text,
        )
        raw_response = chat_completion(
            model=model,
            system_content=prompts.RetrievalEvaluationPrompt.system_prompt,
            human_content=user_prompt,
        )
        # 尝试解析 JSON
        repaired = repair_json(raw_response)
        parsed = json.loads(repaired)
        return ConfidenceScore.from_dict(parsed)
    except Exception as exc:
        logger.warning("评估器调用失败，降级为启发式评分: %s", exc)
        return _heuristic_score(query, docs, answer)
