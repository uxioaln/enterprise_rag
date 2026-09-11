# 构造标准化评估数据集：从已有答案文件中提取 question 与 ground_truth
# 输出 data/eval/eval_dataset.json，供后续批量推理与 RAGAS 评估使用
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

# 定位项目根目录并将其加入 sys.path，便于复用 src 内部模块
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import max_config, base_config  # noqa: E402

# 数据集根目录与评估输出目录
DATA_ROOT = PROJECT_ROOT / "data" / "stock_data"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

# 采样上限：样本数大于该值时随机抽取，保证评估成本可控
SAMPLE_LIMIT = 10
RANDOM_SEED = 42


def _answers_file_candidates() -> list[Path]:
    """生成候选答案文件路径，按优先级排列。

    优先级：answers_max.json -> 对应 max_config 后缀的答案文件 -> answers_base.json
    """
    candidates: list[Path] = []
    # 任务约定名称
    candidates.append(DATA_ROOT / "answers_max.json")
    # max_config 实际生成的答案文件（后缀来自 config_suffix）
    if max_config.config_suffix:
        candidates.append(DATA_ROOT / f"answers{max_config.config_suffix}.json")
    # 兜底：base 配置答案文件
    candidates.append(DATA_ROOT / f"answers{base_config.config_suffix}.json")
    candidates.append(DATA_ROOT / "answers_base.json")
    return candidates


def _resolve_answers_file() -> Optional[Path]:
    """返回第一个存在的候选答案文件，若都不存在返回 None。"""
    for path in _answers_file_candidates():
        if path.exists():
            return path
    return None


def _extract_records(raw: dict) -> list[dict]:
    """从答案文件中提取 (question, ground_truth, kind) 记录列表。

    兼容两种格式：
      1. 提交格式：{"answers": [{"question_text", "kind", "value", ...}]}
      2. 原始格式：{"questions": [{"question_text"/"question", "value"/"answer"/"final_answer", "kind"}]}
    """
    # 优先取 answers 数组，其次 questions 数组
    items = raw.get("answers")
    if not isinstance(items, list):
        items = raw.get("questions")
    if not isinstance(items, list):
        return []

    records: list[dict] = []
    for idx, item in enumerate(items):
        question = item.get("question_text") or item.get("question")
        if not question:
            continue
        # ground_truth 优先级：final_answer -> value -> answer
        ground_truth = (
            item.get("final_answer")
            or item.get("value")
            or item.get("answer")
            or ""
        )
        # 统一转为字符串，避免 None/数值影响后续 RAGAS
        if ground_truth is None:
            ground_truth = ""
        ground_truth = str(ground_truth).strip()
        kind = item.get("kind") or "string"
        records.append({
            "id": idx + 1,
            "question": question,
            "ground_truth": ground_truth,
            "kind": kind,
        })
    return records


def build_dataset(sample_limit: int = SAMPLE_LIMIT, seed: int = RANDOM_SEED) -> Path:
    """构造评估数据集并写出 eval_dataset.json，返回输出路径。"""
    answers_file = _resolve_answers_file()
    if answers_file is None:
        raise FileNotFoundError(
            "未找到答案文件，请先运行 pipeline 生成 answers_*.json（max 或 base 配置）。"
        )

    print(f"[build_eval_dataset] 读取答案文件: {answers_file}")
    with open(answers_file, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = _extract_records(raw)
    if not records:
        raise ValueError(f"未能从 {answers_file} 中提取有效问答记录，请检查文件格式。")

    # 采样：超过上限则随机抽取（固定随机种子保证可复现）
    if len(records) > sample_limit:
        random.seed(seed)
        records = random.sample(records, sample_limit)
        # 重新编号，保证 id 连续
        records = [{**r, "id": i + 1} for i, r in enumerate(records)]
        print(f"[build_eval_dataset] 样本数 {len(records)} 超过上限，随机抽取 {sample_limit} 条")

    # 保证输出目录存在
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_DIR / "eval_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[build_eval_dataset] 评估数据集已生成: {output_path}（共 {len(records)} 条）")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 RAGAS 评估数据集")
    parser.add_argument("--sample-limit", type=int, default=SAMPLE_LIMIT, help="采样上限，默认 10")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="随机采样种子，默认 42")
    args = parser.parse_args()
    build_dataset(sample_limit=args.sample_limit, seed=args.seed)


if __name__ == "__main__":
    main()
