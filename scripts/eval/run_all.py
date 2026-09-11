# 一键式 RAGAS 评估流水线编排：Build -> Generate -> RAGAS -> Analyze
import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
EVAL_SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIGS = "base,pdr,max"


def _run(script_name: str, extra_args: list[str]) -> None:
    """以子进程方式运行 scripts/eval 下的脚本，失败则抛出 RuntimeError 终止编排。"""
    script_path = EVAL_SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)] + extra_args
    print(f"\n{'=' * 60}\n[run_all] 执行: {' '.join(cmd)}\n{'=' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} 执行失败（退出码 {result.returncode}），耗时 {elapsed:.1f} 秒")
    print(f"[run_all] {script_name} 完成，耗时 {elapsed:.1f} 秒")


def main() -> None:
    parser = argparse.ArgumentParser(description="一键运行 RAGAS 评估全流程")
    parser.add_argument(
        "--configs",
        type=str,
        default=DEFAULT_CONFIGS,
        help="待运行的配置，逗号分隔，默认 base,pdr,max",
    )
    args = parser.parse_args()

    overall_t0 = time.time()
    print(f"[run_all] 开始一键评估流水线，配置: {args.configs}")

    # Step 1: 构造评估数据集
    _run("build_eval_dataset.py", [])

    # Step 2: 批量生成答案与上下文
    _run("batch_generate.py", ["--configs", args.configs])

    # Step 3: RAGAS 指标计算
    _run("run_ragas.py", ["--configs", args.configs])

    # Step 4: Bad Case 分析
    _run("analyze_badcase.py", [])

    overall_elapsed = time.time() - overall_t0
    print(f"\n[run_all] 全流程完成，总耗时 {overall_elapsed:.1f} 秒")
    print(f"[run_all] 结果目录: {PROJECT_ROOT / 'data' / 'eval'}")
    print("[run_all] 关键产物：")
    print("  - data/eval/eval_dataset.json    评估数据集")
    print("  - data/eval/results_<config>.json 各配置逐条结果（含 scores）")
    print("  - data/eval/scores.json           聚合分数")
    print("  - data/eval/scores_table.md       Markdown 评分表与结论")
    print("  - data/eval/bad_cases.md          Bad Case 分析报告")


if __name__ == "__main__":
    main()
