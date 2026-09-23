# -*- coding: utf-8 -*-
"""
轻量小样本验证（mini benchmark）：in-process 双臂对照，验证 L1 上下文压缩改造是否有效。

定位：run_benchmark.py 的"轻量前置验证"——不启动 uvicorn 服务、不调用 RAGAS 判官，
只构造两个仅 L1/L2 开关不同的 RunConfig，对少量评测题逐题交错推理
（q1-baseline -> q1-ours -> q2-baseline -> ...，同一题的两臂调用紧邻，
消除 AGICTO 服务端时段差异），直接对比 token 用量与答案内容，
几分钟内给出"改造是否有效"的初步结论。

两臂配置（均基于 max_config 用 replace 派生，仅 L1/L2 开关不同，
其余参数完全一致、检索同一套数据库，保证差异只来自 L1 压缩本身）：
    baseline = enable_l1_compression=False, enable_l2_compression=False（全量上下文）
    ours     = enable_l1_compression=True,  enable_l2_compression=True（L1 问题感知截断）

说明：
    1. 每题独立会话（history=None），L2 历史压缩不触发；多轮长程衰减由
       long_range_stability.py 单独覆盖，本脚本验证的主战场是 L1（A1 改动）；
    2. token 口径与 ab_compare.py 一致：优先服务端真实 prompt_tokens，
       缺失时回退 tiktoken 估算 context_tokens（两臂同用一把尺子）；
    3. 逐题记录 schema 与 batch_generate 的 results_{config}.json 兼容
       （id/question/ground_truth/answer/contexts/token 字段）。

用法（项目根目录、no1 环境下执行，需已配置 .env 与建库）：
    python scripts/eval/mini_bench.py                # 默认 5 题
    python scripts/eval/mini_bench.py --limit 10     # 全量 10 题
    python scripts/eval/mini_bench.py --seed 7       # 换采样种子

输出：
    data/eval/mini_bench_base.json     baseline 臂逐题记录
    data/eval/mini_bench_ours.json     ours 臂逐题记录
    data/eval/mini_bench_report.json   汇总报告（summary + 逐题对比）
    控制台同步打印 token 削减率与逐题答案并排对比（供人眼校验答案质量）

依赖：项目运行环境（不启动 uvicorn、不调用 RAGAS 判官）。
"""

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 定位项目根目录并加入 sys.path（与 scripts/eval 下其他脚本保持一致）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline, max_config  # noqa: E402

# 数据与输出路径（数据源与 batch_generate 保持一致）
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DATA_ROOT = PROJECT_ROOT / "data" / "stock_data"
DATASET_PATH = EVAL_DIR / "eval_dataset.json"

OUT_BASE = EVAL_DIR / "mini_bench_base.json"
OUT_OURS = EVAL_DIR / "mini_bench_ours.json"
OUT_REPORT = EVAL_DIR / "mini_bench_report.json"

# 控制台答案预览的截断长度（完整答案见 JSON 落盘）
_ANSWER_PREVIEW_CHARS = 120


def _load_dataset(limit: int, seed: int) -> list[dict]:
    """读取评测数据集，limit > 0 且样本超限时按固定种子随机采样。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"未找到评估数据集 {DATASET_PATH}，请先运行: python scripts/eval/build_eval_dataset.py"
        )
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    if limit and len(records) > limit:
        total = len(records)
        random.seed(seed)
        records = random.sample(records, limit)
        print(f"[mini_bench] 样本数 {total} 超过上限，随机抽取 {limit} 条（seed={seed}）")
    print(f"[mini_bench] 评测集共 {len(records)} 题")
    return records


def _token_value(rec: dict) -> int | None:
    """单条记录的 token 取值：优先服务端真实 prompt_tokens，缺失回退本地估算。

    与 ab_compare.py 保持同一口径：只要有一侧缺失服务端值，汇总时两臂
    均按各自可得值计算（同臂内部口径一致即可，不影响配对公平性）。
    """
    return rec.get("prompt_tokens") or rec.get("context_tokens")


def _run_one(pipeline: Pipeline, mode_name: str, item: dict) -> dict:
    """单臂单题推理：调用 answer_with_contexts，记录答案、token 用量与耗时。

    单题失败不中断整体流程，记录 error 字段（与 batch_generate 的做法一致）。
    """
    qid = item.get("id")
    rec: dict = {
        "id": qid,
        "question": item.get("question", ""),
        "ground_truth": item.get("ground_truth", ""),
        "kind": item.get("kind", "string"),
        "answer": "",
        "contexts": [],
        "relevant_pages": [],
        "elapsed_seconds": None,
    }
    t0 = time.time()
    try:
        answer_dict, contexts = pipeline.answer_with_contexts(
            item.get("question", ""), kind=item.get("kind", "string"))
        rec["answer"] = str(answer_dict.get("final_answer", ""))
        rec["contexts"] = [str(c) for c in contexts if c]
        rec["relevant_pages"] = answer_dict.get("relevant_pages", [])
        # token 三件套：prompt/completion 为服务端真实值，context 为 tiktoken 估算
        rec["prompt_tokens"] = answer_dict.get("prompt_tokens")
        rec["completion_tokens"] = answer_dict.get("completion_tokens")
        rec["context_tokens"] = answer_dict.get("context_tokens")
    except Exception as err:
        rec["error"] = f"{type(err).__name__}: {err}"
        print(f"[mini_bench] {mode_name} 臂问题 {qid} 失败: {err}")
    rec["elapsed_seconds"] = round(time.time() - t0, 1)
    return rec


def _summarize(recs_base: list[dict], recs_ours: list[dict]) -> dict:
    """汇总双臂统计：token 削减率、时延变化、失败题数。

    配对口径：以 id 配对，任一臂缺 token（失败题）的题剔除，不污染均值；
    削减率 = (baseline_tokens - ours_tokens) / baseline_tokens，正值 = ours 更省。
    """
    base_by_id = {r.get("id"): r for r in recs_base}
    rates: list[float] = []
    bt_all: list[int] = []
    ot_all: list[int] = []
    for o in recs_ours:
        b = base_by_id.get(o.get("id"))
        bt, ot = (_token_value(b) if b else None), _token_value(o)
        if not bt or not ot:
            continue  # 任一臂失败或缺 token 的题跳过
        bt_all.append(bt)
        ot_all.append(ot)
        rates.append((bt - ot) / bt)
    # 时延统计：两臂各自成功题的耗时均值（失败题 elapsed 也已记录但答案为空，剔除）
    bl = [r["elapsed_seconds"] for r in recs_base if r.get("elapsed_seconds") is not None and not r.get("error")]
    ol = [r["elapsed_seconds"] for r in recs_ours if r.get("elapsed_seconds") is not None and not r.get("error")]

    summary: dict = {
        "题数": len(recs_base),
        "有效配对题数": len(rates),
        "baseline 失败题数": sum(1 for r in recs_base if r.get("error")),
        "ours 失败题数": sum(1 for r in recs_ours if r.get("error")),
        "Baseline 平均 tokens": round(statistics.mean(bt_all), 1) if bt_all else None,
        "Ours 平均 tokens": round(statistics.mean(ot_all), 1) if ot_all else None,
        "平均 token 削减率": round(statistics.mean(rates), 4) if rates else None,
        "中位 token 削减率": round(statistics.median(rates), 4) if rates else None,
        "最大 token 削减率": round(max(rates), 4) if rates else None,
        "最小 token 削减率": round(min(rates), 4) if rates else None,
        "Baseline 平均时延_秒": round(statistics.mean(bl), 1) if bl else None,
        "Ours 平均时延_秒": round(statistics.mean(ol), 1) if ol else None,
    }
    # 时延变化率 = (baseline - ours) / baseline，正值 = ours 更快（仅参考，受服务端波动影响大）
    if bl and ol and statistics.mean(bl) != 0:
        summary["时延变化率_正值更快"] = round(
            (statistics.mean(bl) - statistics.mean(ol)) / statistics.mean(bl), 4)
    return summary


def _preview(text: str) -> str:
    """答案预览：截断到固定长度，控制台逐题对比用（完整内容见 JSON）。"""
    text = (text or "").replace("\n", " ").strip()
    return text[:_ANSWER_PREVIEW_CHARS] + ("..." if len(text) > _ANSWER_PREVIEW_CHARS else "")


def _print_per_question(recs_base: list[dict], recs_ours: list[dict]) -> None:
    """控制台逐题并排对比：token、时延、削减率与三份答案预览（人眼校验质量）。"""
    print("\n===== 逐题对比（削减率为正 = ours 更省 token）=====")
    base_by_id = {r.get("id"): r for r in recs_base}
    for o in recs_ours:
        b = base_by_id.get(o.get("id"))
        bt, ot = (_token_value(b) if b else None), _token_value(o)
        rate = f"{(bt - ot) / bt:.1%}" if (bt and ot) else "N/A"
        bl = f"{b['elapsed_seconds']}s" if (b and b.get("elapsed_seconds") is not None) else "N/A"
        ol = f"{o['elapsed_seconds']}s" if o.get("elapsed_seconds") is not None else "N/A"
        print(f"\n[Q{o.get('id')}] base {bt} tok / {bl} | ours {ot} tok / {ol} | 削减率 {rate}")
        print(f"  GT  : {_preview(o.get('ground_truth', ''))}")
        print(f"  Base: {_preview(b.get('answer', '') if b else '')}")
        print(f"  Ours: {_preview(o.get('answer', ''))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="轻量小样本验证：in-process 双臂对照 L1 压缩的 token 削减与答案质量"
    )
    parser.add_argument("--limit", type=int, default=5,
                        help="采样题数上限，0=全量，默认 5（快速验证）")
    parser.add_argument("--seed", type=int, default=42, help="随机采样种子，默认 42")
    args = parser.parse_args()

    # 1) 加载 .env（in-process 推理的 LLM 调用依赖其中的 API Key）
    load_dotenv()

    # 2) 加载评测数据集
    dataset = _load_dataset(args.limit, args.seed)

    # 3) 构造两臂配置：基于 max_config 派生，仅 L1/L2 开关不同，
    #    其余参数（检索、模型、数据库 suffix）完全一致，差异只来自 L1 压缩
    base_cfg = replace(max_config,
                       enable_l1_compression=False,
                       enable_l2_compression=False)
    ours_cfg = replace(max_config,
                       enable_l1_compression=True,
                       enable_l2_compression=True)
    pipe_base = Pipeline(DATA_ROOT, run_config=base_cfg)
    pipe_ours = Pipeline(DATA_ROOT, run_config=ours_cfg)

    # 4) 逐题交错执行：q1-baseline -> q1-ours -> q2-baseline -> ...，
    #    同一题两臂调用紧邻，消除服务端时段差异（方法论与 run_benchmark 一致）
    recs_base: list[dict] = []
    recs_ours: list[dict] = []
    t0 = time.time()
    print(f"\n[mini_bench] ===== 开始交错推理：共 {len(dataset)} 题，每题先 baseline 后 ours =====")
    for item in dataset:
        print(f"\n[mini_bench] Q{item.get('id')} baseline 推理中...")
        rec_b = _run_one(pipe_base, "baseline", item)
        recs_base.append(rec_b)
        print(f"[mini_bench] Q{item.get('id')} ours（L1 开）推理中...")
        rec_o = _run_one(pipe_ours, "ours", item)
        recs_ours.append(rec_o)
    print(f"\n[mini_bench] 交错推理完成，总耗时 {time.time() - t0:.1f} 秒")

    # 5) 汇总统计与逐题并排打印（人眼校验答案质量）
    summary = _summarize(recs_base, recs_ours)
    _print_per_question(recs_base, recs_ours)

    # 6) 三份 JSON 落盘：两臂逐题记录（schema 兼容 batch_generate 生态）+ 汇总报告
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_BASE, "w", encoding="utf-8") as f:
        json.dump(recs_base, f, ensure_ascii=False, indent=2)
    with open(OUT_OURS, "w", encoding="utf-8") as f:
        json.dump(recs_ours, f, ensure_ascii=False, indent=2)
    report = {
        "meta": {
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "题数": len(dataset),
            "采样种子": args.seed,
            "两臂定义": "baseline=L1/L2关（全量上下文），ours=L1/L2开（基于 max_config 仅改开关）",
        },
        "summary": summary,
        "per_question": {
            "baseline": recs_base,
            "ours": recs_ours,
        },
    }
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 7) 控制台汇总输出
    print("\n===== mini_bench 摘要 =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\n[mini_bench] baseline 臂记录: {OUT_BASE}")
    print(f"[mini_bench] ours 臂记录: {OUT_OURS}")
    print(f"[mini_bench] 汇总报告: {OUT_REPORT}")


if __name__ == "__main__":
    main()
