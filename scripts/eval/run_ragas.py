# RAGAS 指标计算：读取三种配置的批量结果，计算四项核心指标并汇总
# 输出：更新 results_{config}.json（追加 scores）、data/eval/scores.json、data/eval/scores_table.md
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 定位项目根目录并加入 sys.path
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = PROJECT_ROOT / "data" / "eval"

# AGICTO 平台 OpenAI 兼容接口
AGICTO_BASE_URL = "https://api.agicto.cn/v1"
# RAGAS 判官模型：使用 gpt-4o-mini（非思维链模型），
# qwen3.8-max 是思维链模型，RAGAS faithfulness/context_precision 的长 prompt 会导致推理超时
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-v4"

# 四项核心评估指标
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
DEFAULT_CONFIGS = ["base", "pdr", "max"]


def _init_evaluators():
    """初始化 RAGAS 评估用的 LLM 与 Embeddings（AGICTO 平台）。"""
    load_dotenv()
    api_key = os.getenv("AGICTO_API_KEY")
    if not api_key:
        raise EnvironmentError("未检测到 AGICTO_API_KEY，请在 .env 中配置。")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    evaluator_llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        base_url=AGICTO_BASE_URL,
        temperature=0,
        timeout=300,       # RAGAS 判官 prompt 较长，放宽超时避免并发下 TimeoutError
        max_retries=2,
    )
    evaluator_embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        base_url=AGICTO_BASE_URL,
        # 关闭 tiktoken 长度检查：默认会把文本转为 token ID 整数数组发送，
        # AGICTO 不支持该输入格式（返回空数据），关闭后直接发送原始字符串
        check_embedding_ctx_length=False,
        chunk_size=10,     # AGICTO text-embedding-v4 服务端限制单批不超过 10 条，超限返回空数据
        timeout=300,
        max_retries=2,
    )
    return evaluator_llm, evaluator_embeddings


def _load_results(config_name: str) -> Optional[list[dict]]:
    """读取某个配置的批量结果，不存在则返回 None。"""
    path = EVAL_DIR / f"results_{config_name}.json"
    if not path.exists():
        print(f"[run_ragas] 跳过 {config_name}：未找到 {path}（请先运行 batch_generate）")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_dataset(records: list[dict]):
    """从记录构建 RAGAS EvaluationDataset，返回 (dataset, valid_indices)。"""
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    samples = []
    valid_indices: list[int] = []
    for i, rec in enumerate(records):
        answer = (rec.get("answer") or "").strip()
        contexts = rec.get("contexts") or []
        # 过滤无效记录：无答案或无上下文无法参与指标计算
        if not answer or not contexts:
            continue
        # contexts 必须为字符串列表
        ctx_texts = [str(c) for c in contexts if c]
        if not ctx_texts:
            continue
        samples.append(
            SingleTurnSample(
                user_input=rec.get("question", ""),
                response=answer,
                retrieved_contexts=ctx_texts,
                reference=str(rec.get("ground_truth", "") or ""),
            )
        )
        valid_indices.append(i)
    dataset = EvaluationDataset(samples=samples)
    return dataset, valid_indices


def _run_metrics(config_name: str, records: list[dict], llm, embeddings) -> list[dict]:
    """对单个配置计算四项指标，返回带 scores 字段的记录列表。"""
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    dataset, valid_indices = _build_dataset(records)
    print(f"[run_ragas] 配置 {config_name}：{len(valid_indices)}/{len(records)} 条有效记录参与计算")
    if not valid_indices:
        print(f"[run_ragas] 配置 {config_name} 无有效记录，跳过指标计算")
        return records

    print(f"[run_ragas] 配置 {config_name} 开始 RAGAS 评估（可能耗时较长）...")
    t0 = time.time()
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
        raise_exceptions=False,
    )
    elapsed = time.time() - t0
    print(f"[run_ragas] 配置 {config_name} RAGAS 评估完成，耗时 {elapsed:.1f} 秒")

    # 将逐行分数回填到记录
    df = result.to_pandas()
    # 取出指标列，按 valid_indices 顺序回填
    for offset, rec_idx in enumerate(valid_indices):
        row = df.iloc[offset]
        # 合并策略：本轮某指标失败（None/NaN）时保留该记录上一轮已有的有效分数，
        # 避免配额不足、超时等部分失败在重跑时把已成功的指标覆盖为空
        existing_scores = records[rec_idx].get("scores") or {}
        scores = {}
        for metric in METRIC_NAMES:
            val = None
            if metric in row:
                val = row[metric]
                # 处理 NaN
                try:
                    val = float(val)
                    if val != val:  # NaN
                        val = None
                except (TypeError, ValueError):
                    val = None
            scores[metric] = val if val is not None else existing_scores.get(metric)
        records[rec_idx]["scores"] = scores
    return records


def _aggregate(records: list[dict]) -> dict:
    """计算单配置的四项指标均值（忽略 None）。"""
    sums = {m: 0.0 for m in METRIC_NAMES}
    counts = {m: 0 for m in METRIC_NAMES}
    for rec in records:
        scores = rec.get("scores")
        if not scores:
            continue
        for m in METRIC_NAMES:
            v = scores.get(m)
            if v is not None:
                sums[m] += v
                counts[m] += 1
    return {
        m: round(sums[m] / counts[m], 4) if counts[m] else None
        for m in METRIC_NAMES
    }


def _build_scores_table(agg: dict, config_names: list[str]) -> str:
    """生成 Markdown 表格与简短结论。"""
    header = "| 配置 | " + " | ".join(METRIC_NAMES) + " |"
    sep = "|---|" + "|".join(["---"] * len(METRIC_NAMES)) + "|"
    lines = [header, sep]
    for name in config_names:
        if name not in agg:
            continue
        row_scores = agg[name]
        cells = []
        for m in METRIC_NAMES:
            v = row_scores.get(m)
            cells.append(f"{v:.2f}" if v is not None else "N/A")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    # 简短结论：找出每项指标最优配置
    conclusion_lines = ["", "### 结论", ""]
    for m in METRIC_NAMES:
        best_cfg = None
        best_val = -1.0
        for name in config_names:
            v = agg.get(name, {}).get(m)
            if v is not None and v > best_val:
                best_val = v
                best_cfg = name
        if best_cfg is not None:
            conclusion_lines.append(f"- {m}：`{best_cfg}` 配置表现最佳（{best_val:.2f}）")
        else:
            conclusion_lines.append(f"- {m}：无有效数据")
    return "\n".join(lines + conclusion_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="计算 RAGAS 四项核心指标并汇总")
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(DEFAULT_CONFIGS),
        help="待评估的配置，逗号分隔，默认 base,pdr,max",
    )
    args = parser.parse_args()

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    llm, embeddings = _init_evaluators()

    agg: dict[str, dict] = {}
    for name in config_names:
        records = _load_results(name)
        if records is None:
            continue
        try:
            records = _run_metrics(name, records, llm, embeddings)
        except Exception as err:
            print(f"[run_ragas] 配置 {name} 评估失败: {err}")
        # 回写带 scores 的结果文件
        out_path = EVAL_DIR / f"results_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        agg[name] = _aggregate(records)

    # 汇总 scores.json
    scores_path = EVAL_DIR / "scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"[run_ragas] 聚合分数已写入: {scores_path}")

    # 汇总 scores_table.md
    table_md = _build_scores_table(agg, config_names)
    table_path = EVAL_DIR / "scores_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# RAGAS 评估分数汇总\n\n")
        f.write(table_md)
    print(f"[run_ragas] Markdown 表格已写入: {table_path}")


if __name__ == "__main__":
    main()
