# Bad Case 分析：筛选失败案例，深度诊断根因，输出 data/eval/bad_cases.md
import json
import sys
from pathlib import Path
from typing import Optional

# 定位项目根目录并加入 sys.path
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EVAL_DIR = PROJECT_ROOT / "data" / "eval"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Bad case 判定阈值
THRESHOLDS = {
    "faithfulness": 0.6,
    "answer_relevancy": 0.6,
    "context_precision": 0.5,
    "context_recall": 0.5,
}
DEFAULT_CONFIGS = ["base", "pdr", "max"]
MAX_CASES = 5
TRUNCATE_LEN = 200
CONTEXT_CHUNKS_SHOWN = 3


def _load_results(config_name: str) -> Optional[list[dict]]:
    path = EVAL_DIR / f"results_{config_name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_bad_case(rec: dict) -> bool:
    """根据阈值判定是否为 bad case；含 error 或缺失分数亦视为 bad case。"""
    if rec.get("error"):
        return True
    scores = rec.get("scores")
    if not scores:
        return True
    for metric, threshold in THRESHOLDS.items():
        v = scores.get(metric)
        if v is not None and v < threshold:
            return True
    return False


def _diagnose(rec: dict) -> tuple[str, str]:
    """根据分数与状态给出根因标签与详细分析。"""
    if rec.get("error"):
        return "检索失败", f"推理阶段抛出异常：{rec.get('error')}，未能生成答案或上下文。"

    scores = rec.get("scores") or {}
    # 找出最差的指标作为主因
    worst_metric = None
    worst_val = None
    for m in METRIC_NAMES:
        v = scores.get(m)
        if v is not None and (worst_val is None or v < worst_val):
            worst_val = v
            worst_metric = m

    label_map = {
        "context_recall": "检索漏召",
        "context_precision": "重排错位",
        "faithfulness": "幻觉",
        "answer_relevancy": "答案偏题",
    }
    label = label_map.get(worst_metric, "其他")

    detail_parts = []
    for m in METRIC_NAMES:
        v = scores.get(m)
        detail_parts.append(f"{m}={v:.2f}" if v is not None else f"{m}=N/A")

    if worst_metric == "context_recall":
        detail = (
            f"上下文召回不足（context_recall={worst_val:.2f}），ground_truth 中的关键信息未被检索到；"
            f"可能原因：分块粒度过大/过小、嵌入模型语义匹配偏差、或问题与文档表述差异较大。"
            f"指标明细：{', '.join(detail_parts)}。"
        )
    elif worst_metric == "context_precision":
        detail = (
            f"上下文精度偏低（context_precision={worst_val:.2f}），检索结果中混入较多无关 chunk；"
            f"可能原因：LLM 重排未生效或 top_n 过大引入噪声。"
            f"指标明细：{', '.join(detail_parts)}。"
        )
    elif worst_metric == "faithfulness":
        detail = (
            f"答案忠实度不足（faithfulness={worst_val:.2f}），生成内容存在超出上下文支撑的表述（幻觉）；"
            f"可能原因：模型过度发散或上下文信息不足导致补全。"
            f"指标明细：{', '.join(detail_parts)}。"
        )
    elif worst_metric == "answer_relevancy":
        detail = (
            f"答案相关性不足（answer_relevancy={worst_val:.2f}），答案未紧扣问题；"
            f"可能原因：问题表述复杂或检索上下文引导偏离。"
            f"指标明细：{', '.join(detail_parts)}。"
        )
    else:
        detail = f"多项指标偏低，需综合排查。指标明细：{', '.join(detail_parts)}。"
    return label, detail


def _truncate(text: str, n: int = TRUNCATE_LEN) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n] + "..."


def _select_cases(all_bad: list[dict], max_cases: int = MAX_CASES) -> list[dict]:
    """挑选代表性案例：兼顾配置与根因多样性，按严重程度排序。"""
    # 严重程度 = 最低分数（越低越严重），无分数视为 0
    def severity(rec: dict) -> float:
        scores = rec.get("scores") or {}
        vals = [v for v in scores.values() if v is not None]
        return min(vals) if vals else 0.0

    all_bad.sort(key=severity)
    selected: list[dict] = []
    seen_configs: set[str] = set()
    seen_labels: set[str] = set()
    # 第一轮：优先选取不同配置、不同根因的案例
    for rec in all_bad:
        label, _ = _diagnose(rec)
        cfg = rec["config"]
        if cfg in seen_configs and label in seen_labels:
            continue
        selected.append(rec)
        seen_configs.add(cfg)
        seen_labels.add(label)
        if len(selected) >= max_cases:
            break
    # 第二轮：若不足，补充剩余严重案例
    for rec in all_bad:
        if rec in selected:
            continue
        selected.append(rec)
        if len(selected) >= max_cases:
            break
    return selected[:max_cases]


def _render_case(rec: dict) -> str:
    label, detail = _diagnose(rec)
    scores = rec.get("scores") or {}
    score_str = ", ".join(
        f"{m}={scores.get(m):.2f}" if scores.get(m) is not None else f"{m}=N/A"
        for m in METRIC_NAMES
    )
    contexts = rec.get("contexts") or []
    ctx_summary = "\n\n".join(
        f"- chunk{i+1}: {_truncate(c, 150)}" for i, c in enumerate(contexts[:CONTEXT_CHUNKS_SHOWN])
    ) or "- （无上下文）"

    lines = [
        f"### 案例 {rec['id']} [{rec['config']}] — 根因：{label}",
        "",
        f"- **问题**：{rec.get('question', '')}",
        f"- **分数**：{score_str}",
        f"- **相关页面**：{rec.get('relevant_pages', [])}",
        "",
        "**内容对比**：",
        "",
        f"- **Ground Truth（截断 {TRUNCATE_LEN} 字）**：{_truncate(rec.get('ground_truth', ''))}",
        f"- **Generated Answer（截断 {TRUNCATE_LEN} 字）**：{_truncate(rec.get('answer', ''))}",
        f"- **上下文摘要（前 {CONTEXT_CHUNKS_SHOWN} 个 chunk）**：",
        "",
        ctx_summary,
        "",
        f"**诊断**：{label}",
        "",
        detail,
        "",
    ]
    return "\n".join(lines)


def _build_suggestions(cases: list[dict]) -> str:
    """基于共性给出优化建议。"""
    label_counts: dict[str, int] = {}
    for rec in cases:
        label, _ = _diagnose(rec)
        label_counts[label] = label_counts.get(label, 0) + 1

    suggestions = ["## 优化建议（基于共性根因）", ""]
    suggestion_map = {
        "检索漏召": "调小分块粒度或扩大 top_n，必要时引入 BM25 混合检索提升召回；检查嵌入模型是否与中文研报语义匹配。",
        "重排错位": "启用 LLM 重排（max 配置），调小 top_n 候选数，剔除低相关 chunk。",
        "幻觉": "在 prompt 中强化“仅基于上下文作答”约束，降低模型温度，必要时改用支持结构化输出的更强模型。",
        "答案偏题": "优化问题路由与问题改写，确保检索 query 与用户意图一致。",
        "检索失败": "检查对应配置的向量库是否已构建（databases_<suffix>/vector_dbs），确认 subset.csv 中公司名与文档 metainfo 一致。",
    }
    for label in sorted(label_counts, key=lambda x: label_counts[x], reverse=True):
        suggestions.append(f"- **{label}**（{label_counts[label]} 例）：{suggestion_map.get(label, '综合排查。')}")
    return "\n".join(suggestions) + "\n"


def main() -> None:
    all_bad: list[dict] = []
    for name in DEFAULT_CONFIGS:
        records = _load_results(name)
        if records is None:
            print(f"[analyze_badcase] 跳过 {name}：未找到结果文件")
            continue
        for rec in records:
            rec = dict(rec)
            rec["config"] = name
            if _is_bad_case(rec):
                all_bad.append(rec)

    print(f"[analyze_badcase] 共筛选出 {len(all_bad)} 个 bad case")
    if not all_bad:
        print("[analyze_badcase] 无 bad case，跳过报告生成")
        return

    selected = _select_cases(all_bad)
    print(f"[analyze_badcase] 选取 {len(selected)} 个代表性案例进行深度分析")

    md_lines = [
        "# Bad Case 分析报告",
        "",
        f"共筛选 bad case {len(all_bad)} 个，下文展示 {len(selected)} 个代表性案例。",
        "",
        "判定阈值：faithfulness < 0.6 或 answer_relevancy < 0.6 或 context_precision < 0.5 或 context_recall < 0.5。",
        "",
    ]
    for rec in selected:
        md_lines.append(_render_case(rec))
    md_lines.append(_build_suggestions(selected))

    out_path = EVAL_DIR / "bad_cases.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[analyze_badcase] 报告已生成: {out_path}")


if __name__ == "__main__":
    main()
