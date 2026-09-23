# -*- coding: utf-8 -*-
"""
信息密度 (Information Density) 指标计算：黄金答案Token数 / 平均上下文Token数。

含义：每 1 个上下文 Token 中承载了多少"黄金答案级"的有效信息。
上下文压缩（L1 裁剪 / L2 摘要）后密度上升，说明压缩去除了冗余而保留了有效信息。

用法：
    python scripts/eval/info_density.py --configs baseline,ours

输入：
    data/eval/results_{config}.json  批量推理结果（batch_generate 产物）

输出：
    data/eval/info_density_{config}.json   单配置明细（summary + per_question）
    data/eval/info_density_summary.json    多配置汇总；恰好两配置时附带 A/B 差值

计算口径（两种同时输出，便于交叉验证）：
    口径A（纯上下文，主口径）：对 rec["contexts"] 逐条 tiktoken 计数后求和，
                              只统计检索上下文本身的 token；
    口径B（落盘交叉验证）    ：直接取 rec["context_tokens"]（批量推理时落盘的
                              system + user 整体 prompt 估算值）。
    token 编码统一 o200k_base，与 BaseOpenaiProcessor.count_tokens 口径一致。

密度定义：
    总体信息密度 = 平均黄金答案Token / 平均上下文Token（公式的字面语义）；
    逐题密度     = 该题黄金Token / 该题上下文Token，用于分布统计（均值/中位/P95）。

依赖：仅 tiktoken + Python 标准库，无 LLM 调用，可独立运行。
"""

import argparse
import json
import statistics
from pathlib import Path

import tiktoken

# 评估输出目录（定位项目根目录，与 scripts/eval 下其他脚本保持一致）
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "eval"

# 默认对比的两臂：A/B 压缩实验（与 ab_compare.py 默认一致）
DEFAULT_CONFIGS = ["baseline", "ours"]

# token 编码：与 BaseOpenaiProcessor.count_tokens 的默认编码保持同一口径
_ENCODING_NAME = "o200k_base"
# 模块级编码器（tiktoken 内部有缓存，重复获取无额外开销）
_ENCODER = tiktoken.get_encoding(_ENCODING_NAME)


def _count_tokens(text: str) -> int:
    """统计字符串 token 数，口径与 BaseOpenaiProcessor.count_tokens 一致（o200k_base）。"""
    return len(_ENCODER.encode(text))


def _load_records(config: str) -> list[dict]:
    """读取某个配置的批量推理结果，文件不存在时给出清晰报错。"""
    path = EVAL_DIR / f"results_{config}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {path}，请先运行: python scripts/eval/batch_generate.py --configs {config}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _per_question_density(records: list[dict]) -> tuple[list[dict], int]:
    """逐题计算信息密度，返回 (明细行列表, 跳过题数)。

    逐题密度（口径A）= 黄金答案Token数 / 该题上下文Token总数。
    黄金答案为空或上下文为空的题密度无意义，跳过不污染统计
    （与 ab_compare 跳过缺 token 记录的做法一致）。
    """
    rows: list[dict] = []
    skipped = 0
    for rec in records:
        gt = (rec.get("ground_truth") or "").strip()
        # contexts 统一转为非空字符串列表（与 run_ragas 的处理一致）
        contexts = [str(c) for c in (rec.get("contexts") or []) if c]
        if not gt or not contexts:
            skipped += 1
            continue
        # 黄金答案 token 数
        gt_tokens = _count_tokens(gt)
        # 口径A：逐条计数上下文 chunk 后求和（纯上下文 token）
        ctx_tokens = sum(_count_tokens(c) for c in contexts)
        if gt_tokens == 0 or ctx_tokens == 0:
            skipped += 1
            continue
        # 口径B：批量推理时落盘的整体 prompt token 估算（可能缺失）
        ctx_recorded = rec.get("context_tokens")
        rows.append({
            "id": rec.get("id"),
            "question": (rec.get("question") or "")[:40],
            "gt_tokens": gt_tokens,
            "ctx_tokens": ctx_tokens,
            "ctx_tokens_recorded": ctx_recorded,
            # 口径A逐题密度：黄金Token / 纯上下文Token
            "density": round(gt_tokens / ctx_tokens, 6),
            # 口径B逐题密度：黄金Token / 落盘整体prompt估算（缺失时为 None）
            "density_recorded": round(gt_tokens / ctx_recorded, 6) if ctx_recorded else None,
        })
    return rows, skipped


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    """简单百分位计算：取排序后索引 p*n 位置的值（避免引入第三方库）。"""
    if not sorted_vals:
        return None
    idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
    return sorted_vals[idx]


def _summarize(config: str, rows: list[dict], total: int, skipped: int) -> dict:
    """汇总单配置的信息密度统计：口径A为主口径，口径B交叉验证。"""
    gt_all = [r["gt_tokens"] for r in rows]
    ctx_all = [r["ctx_tokens"] for r in rows]
    dens_all = [r["density"] for r in rows]
    sorted_dens = sorted(dens_all)

    summary: dict = {
        "config": config,
        "总题数": total,
        "有效样本数": len(rows),
        "跳过题数": skipped,
        "平均黄金答案Token": round(statistics.mean(gt_all), 1),
        "平均上下文Token_口径A纯上下文": round(statistics.mean(ctx_all), 1),
        # 总体信息密度（用户公式）：平均黄金Token / 平均上下文Token
        "总体信息密度_口径A": round(statistics.mean(gt_all) / statistics.mean(ctx_all), 6),
        # 逐题密度的分布统计（口径A）
        "逐题密度均值_口径A": round(statistics.mean(dens_all), 6),
        "逐题密度中位数_口径A": round(statistics.median(dens_all), 6),
        "逐题密度P95_口径A": round(_percentile(sorted_dens, 0.95), 6),
        "逐题密度最大值_口径A": round(max(dens_all), 6),
        "逐题密度最小值_口径A": round(min(dens_all), 6),
    }

    # 口径B：仅统计有落盘 context_tokens 的题（分子分母保持同一批样本）
    rows_b = [r for r in rows if r["ctx_tokens_recorded"]]
    if rows_b:
        gt_b = [r["gt_tokens"] for r in rows_b]
        ctx_b = [r["ctx_tokens_recorded"] for r in rows_b]
        summary["平均上下文Token_口径B落盘prompt"] = round(statistics.mean(ctx_b), 1)
        summary["总体信息密度_口径B"] = round(statistics.mean(gt_b) / statistics.mean(ctx_b), 6)
    return summary


# 两配置对比时输出差值的核心指标（口径A为主）
_DIFF_KEYS = [
    "总体信息密度_口径A",
    "逐题密度均值_口径A",
    "平均上下文Token_口径A纯上下文",
    "平均黄金答案Token",
]


def _build_diff(s_a: dict, s_b: dict, name_a: str, name_b: str) -> dict:
    """两配置差值（后一配置 - 前一配置），用于 A/B 压缩实验解读。

    密度类指标 diff 为正 = 后一配置（如 ours）密度更高，压缩去冗余保有效；
    上下文 Token 类指标 diff 为负 = 后一配置更省。
    """
    diff: dict = {"对比说明": f"{name_b} - {name_a}（密度类正值=后者更高，Token类负值=后者更省）"}
    for key in _DIFF_KEYS:
        va, vb = s_a.get(key), s_b.get(key)
        diff[key] = {
            name_a: va,
            name_b: vb,
            "diff": round(vb - va, 6) if (va is not None and vb is not None) else None,
        }
    return diff


def main() -> None:
    parser = argparse.ArgumentParser(description="信息密度计算：黄金答案Token数 / 平均上下文Token数")
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_CONFIGS),
        help=f"待计算的配置，逗号分隔，默认 {','.join(DEFAULT_CONFIGS)}",
    )
    args = parser.parse_args()
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    # 1) 逐配置计算信息密度并落盘明细
    summaries: dict[str, dict] = {}
    for name in config_names:
        records = _load_records(name)
        rows, skipped = _per_question_density(records)
        print(f"[info_density] 配置 {name}: 有效 {len(rows)} 题 / 跳过 {skipped} 题 / 共 {len(records)} 题")
        if not rows:
            print(f"[info_density] 配置 {name} 无有效样本（黄金答案或上下文为空），跳过")
            continue
        summary = _summarize(name, rows, len(records), skipped)
        summaries[name] = summary
        # 单配置明细落盘（summary + per_question）
        out_path = EVAL_DIR / f"info_density_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_question": rows}, f, ensure_ascii=False, indent=2)
        print(f"[info_density] 配置 {name} 明细已写入: {out_path}")

    if not summaries:
        raise RuntimeError("所有配置均无有效样本（黄金答案或上下文为空），无法计算信息密度")

    # 2) 多配置汇总；恰好两配置时附带 A/B 差值（如 baseline,ours）
    report: dict = {"summary": summaries}
    names = list(summaries)
    if len(names) == 2:
        report["对比"] = _build_diff(summaries[names[0]], summaries[names[1]], names[0], names[1])
    summary_path = EVAL_DIR / "info_density_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 3) 控制台摘要输出
    print("\n===== 信息密度摘要（口径A：纯上下文；口径B见 JSON 报告）=====")
    for name, s in summaries.items():
        print(
            f"{name}: 总体密度={s['总体信息密度_口径A']} 逐题密度均值={s['逐题密度均值_口径A']} "
            f"平均黄金Token={s['平均黄金答案Token']} 平均上下文Token={s['平均上下文Token_口径A纯上下文']}"
        )
    if "对比" in report:
        print(f"\n===== A/B 差值（{names[1]} - {names[0]}）=====")
        for key, d in report["对比"].items():
            if key == "对比说明":
                continue
            print(f"{key}: {names[0]}={d[names[0]]} {names[1]}={d[names[1]]} diff={d['diff']}")
    print(f"\n[info_density] 汇总报告已写入: {summary_path}")


if __name__ == "__main__":
    main()
