# 批量生成答案与上下文：针对 base/pdr/max 三种配置，逐条调用 Pipeline.answer_with_contexts
# 输出 data/eval/results_{config}.json，供后续 RAGAS 指标计算使用
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm

# 定位项目根目录并加入 sys.path
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline, configs  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "data" / "stock_data"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DATASET_PATH = EVAL_DIR / "eval_dataset.json"

# 默认运行的三种配置
DEFAULT_CONFIGS = ["base", "pdr", "max"]


def _load_dataset() -> list[dict]:
    """读取评估数据集，若不存在则提示先运行 build_eval_dataset。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"未找到评估数据集 {DATASET_PATH}，请先运行: python scripts/eval/build_eval_dataset.py"
        )
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_config(config_name: str, dataset: list[dict]) -> Path:
    """对单一配置批量推理，写出 results_{config}.json，返回输出路径。"""
    run_config = configs[config_name]
    print(f"\n[batch_generate] 配置: {config_name} (suffix={run_config.config_suffix})")
    pipeline = Pipeline(DATA_ROOT, run_config=run_config)

    results: list[dict] = []
    for item in tqdm(dataset, desc=f"[{config_name}] 生成答案"):
        qid = item.get("id")
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        kind = item.get("kind", "string")
        record = {
            "id": qid,
            "question": question,
            "ground_truth": ground_truth,
            "kind": kind,
        }
        try:
            answer_dict, contexts = pipeline.answer_with_contexts(question, kind=kind)
            record["answer"] = str(answer_dict.get("final_answer", ""))
            record["contexts"] = contexts
            record["relevant_pages"] = answer_dict.get("relevant_pages", [])
            record["references"] = answer_dict.get("references", [])
            record["step_by_step_analysis"] = answer_dict.get("step_by_step_analysis", "")
            record["reasoning_summary"] = answer_dict.get("reasoning_summary", "")
        except Exception as err:
            # 单条失败不中断整体流程，记录错误信息
            record["answer"] = ""
            record["contexts"] = []
            record["relevant_pages"] = []
            record["error"] = f"{type(err).__name__}: {err}"
            record["traceback"] = traceback.format_exc()
            print(f"\n[batch_generate] 配置 {config_name} 问题 {qid} 失败: {err}")
        results.append(record)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_DIR / f"results_{config_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[batch_generate] 配置 {config_name} 完成 -> {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="批量生成三种配置的答案与上下文")
    parser.add_argument(
        "--configs",
        type=str,
        default=",".join(DEFAULT_CONFIGS),
        help="待运行的配置，逗号分隔，默认 base,pdr,max",
    )
    args = parser.parse_args()

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    invalid = [c for c in config_names if c not in configs]
    if invalid:
        raise ValueError(f"未知配置: {invalid}，可用配置: {list(configs.keys())}")

    dataset = _load_dataset()
    print(f"[batch_generate] 评估数据集: {len(dataset)} 条问题")

    t0 = time.time()
    for name in config_names:
        _run_config(name, dataset)
    elapsed = time.time() - t0
    print(f"\n[batch_generate] 全部配置完成，总耗时 {elapsed:.1f} 秒")
    print(f"[batch_generate] 结果目录: {EVAL_DIR}")


if __name__ == "__main__":
    main()
