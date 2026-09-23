# -*- coding: utf-8 -*-
"""
A/B Test 聚合分析：计算实验组（Ours）相对对照组（Baseline）的 prompt_tokens 削减率。

用法：
    python scripts/eval/ab_compare.py --baseline baseline --ours ours

输入：
    data/eval/results_{baseline}.json  对照组批量推理结果（batch_generate 产物）
    data/eval/results_{ours}.json      实验组批量推理结果（batch_generate 产物）

输出：
    data/eval/ab_compare_report.json   聚合报告（summary + ragas_diff + per_question）

主指标：
    削减率 = (Baseline_tokens - Ours_tokens) / Baseline_tokens，正值表示 Ours 更省
    token 取值优先级：服务端真实 prompt_tokens > 本地 tiktoken 估算 context_tokens
    （两臂必须用同一把尺子：只要有一侧缺失服务端值，该题整体回退估算口径）

辅助指标：
    RAGAS 四项指标差值（Ours - Baseline），用于佐证压缩未损质量；
    任一臂未跑 run_ragas（无 scores 字段）时自动跳过该部分。

依赖：仅 Python 标准库，可独立运行。
"""

import argparse
import json
import statistics
from pathlib import Path

# 定位项目根目录并加入 sys.path（与 scripts/eval 下其他脚本保持一致）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

# RAGAS 四项核心指标名（与 run_ragas.py 保持一致）
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _load_records(config: str) -> list[dict]:
    """读取某个配置的批量推理结果，文件不存在时给出清晰报错。"""
    path = EVAL_DIR / f"results_{config}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {path}，请先运行: python scripts/eval/batch_generate.py --configs {config}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _token_value(rec: dict) -> int | None:
    """单条记录的 token 取值：优先服务端真实 prompt_tokens，缺失则回退本地估算。"""
    return rec.get("prompt_tokens") or rec.get("context_tokens")


def _per_question_delta(base: list[dict], ours: list[dict]) -> tuple[list[dict], int, int]:
    """逐题配对计算削减率。

    返回 (明细行列表, 有效配对数, 总题数)。
    配对键为题目 id；任一臂 token 缺失（如配额失败/超时）的题跳过，不污染均值。
    """
    base_by_id = {r.get("id"): r for r in base}
    rows: list[dict] = []
    for o in ours:
        b = base_by_id.get(o.get("id"))
        bt, ot = (_token_value(b) if b else None), _token_value(o)
        if not bt or not ot:
            continue  # 缺 token 的题跳过（失败记录 answer 为空且无 token 字段）
        # 口径标注：两侧均有服务端真实值 -> server；否则为估算口径
        source = "server" if (b.get("prompt_tokens") and o.get("prompt_tokens")) else "estimated"
        rows.append({
            "id": o.get("id"),
            "question": (o.get("question") or "")[:40],
            "baseline_tokens": bt,
            "ours_tokens": ot,
            "token_source": source,
            # 正值 = Ours 更省，负值 = Ours 反而更长
            "reduction_rate": round((bt - ot) / bt, 4),
        })
    return rows, len(rows), len(ours)


def _ragas_diff(base: list[dict], ours: list[dict]) -> dict | None:
    """计算 RAGAS 四项指标的均值差（Ours - Baseline），用于质量佐证。

    任一臂没有任何 scores 字段时返回 None（表示未跑 run_ragas，跳过）。
    """
    def _avg(records: list[dict], metric: str) -> float | None:
        vals = [r.get("scores", {}).get(metric) for r in records]
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    base_scored = any(r.get("scores") for r in base)
    ours_scored = any(r.get("scores") for r in ours)
    if not (base_scored and ours_scored):
        return None

    diff: dict[str, dict] = {}
    for m in RAGAS_METRICS:
        b_avg, o_avg = _avg(base, m), _avg(ours, m)
        diff[m] = {
            "baseline": round(b_avg, 4) if b_avg is not None else None,
            "ours": round(o_avg, 4) if o_avg is not None else None,
            # 正值 = Ours 质量指标更高；None 表示某臂缺该指标数据
            "diff": round(o_avg - b_avg, 4) if (b_avg is not None and o_avg is not None) else None,
        }
    return diff


def _percentile(sorted_rates: list[float], p: float) -> float | None:
    """简单百分位计算：取排序后索引 p*n 位置的值（避免引入第三方库）。"""
    if not sorted_rates:
        return None
    idx = min(int(len(sorted_rates) * p), len(sorted_rates) - 1)
    return sorted_rates[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B Test 聚合：prompt_tokens 削减率与 RAGAS 差值")
    parser.add_argument("--baseline", default="baseline", help="对照组配置名，默认 baseline")
    parser.add_argument("--ours", default="ours", help="实验组配置名，默认 ours")
    args = parser.parse_args()

    # 1) 读取两臂批量推理结果
    base = _load_records(args.baseline)
    ours = _load_records(args.ours)
    print(f"[ab_compare] Baseline({args.baseline}): {len(base)} 条 | Ours({args.ours}): {len(ours)} 条")

    # 2) 逐题配对计算削减率
    rows, valid_n, total_n = _per_question_delta(base, ours)
    if not rows:
        raise RuntimeError(
            f"无有效配对样本（Baseline {len(base)} 条 / Ours {len(ours)} 条），"
            "请检查两臂 results 文件是否来自同一评估数据集且 token 字段完整"
        )

    # 3) 汇总统计
    rates = [r["reduction_rate"] for r in rows]
    sorted_rates = sorted(rates)
    bt_all = [r["baseline_tokens"] for r in rows]
    ot_all = [r["ours_tokens"] for r in rows]
    server_n = sum(1 for r in rows if r["token_source"] == "server")

    summary = {
        "baseline_config": args.baseline,
        "ours_config": args.ours,
        "有效样本数": valid_n,
        "总题数": total_n,
        "服务端真实口径题数": server_n,
        "Baseline 平均 prompt_tokens": round(statistics.mean(bt_all), 1),
        "Ours 平均 prompt_tokens": round(statistics.mean(ot_all), 1),
        "平均削减率 (Baseline-Ours)/Baseline": round(statistics.mean(rates), 4),
        "中位削减率": round(statistics.median(rates), 4),
        "P95 削减率": round(_percentile(sorted_rates, 0.95), 4),
        "最大削减率": round(max(rates), 4),
        "最小削减率": round(min(rates), 4),
    }

    # 4) RAGAS 质量佐证（未跑 run_ragas 时为 None）
    ragas_diff = _ragas_diff(base, ours)

    # 5) 写出聚合报告
    report = {"summary": summary, "ragas_diff": ragas_diff, "per_question": rows}
    out_path = EVAL_DIR / "ab_compare_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 6) 控制台摘要输出
    print("\n===== A/B Test 摘要 =====")
    for k, v in summary.items():
        print(f"{k}: {v}")
    if ragas_diff:
        print("\n===== RAGAS 质量差值（Ours - Baseline，正值=Ours 更优）=====")
        for m, d in ragas_diff.items():
            print(f"{m}: baseline={d['baseline']} ours={d['ours']} diff={d['diff']}")
    else:
        print("\n[ab_compare] 未检测到 RAGAS scores，跳过质量差值（可先运行 run_ragas.py）")
    print(f"\n[ab_compare] 报告已写入: {out_path}")


if __name__ == "__main__":
    main()
