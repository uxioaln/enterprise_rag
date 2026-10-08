# -*- coding: utf-8 -*-
"""
检索-only 召回率评测脚本：不调用生成 LLM，直接测量各检索臂的 Top-k 页码召回率。

用途：补齐 README"评测结果"表中 Top-5 召回率一栏的复现口径。
  - baseline 臂（vector）：单路向量检索（FAISS，llm_reranking=False）
  - ours 臂（hybrid）：向量候选 + LLM 重排（HybridRetriever，llm_reranking=True，
    即 /chat 服务链路"可开关混合检索 + LLM 重排"的实际实现；
    注：BM25Retriever 未接入单题检索路径，单题链路的"混合"实为向量+重排）

设计要点：
  1. 检索路径与 /chat 完全同源：in-process 构造 Pipeline，逐题调用
     QuestionsProcessor.retrieve_question_contexts（与 Pipeline._retrieve_contexts
     内部调用一致，含公司路由与多公司聚合），跳过生成阶段，无判官噪声；
  2. 两臂仅 llm_reranking 一个开关不同（parent_document_retrieval 同为 True，
     返回父页面并按页去重，与"答案可溯源到页码"的口径一致），保证差异可归因；
  3. 金标准为人工标注的页码（gold_pages.json），题目命中定义：
     金标准中每家公司各自的前 k 条检索结果里出现任一金标准页码即该公司命中，
     题目级命中 = 所有金标准公司均命中（单公司题退化为普通 hit@k）；
  4. 结果 -> 公司的归属通过 file_name -> metainfo.company_name 映射判定；
     单一金标准公司时容忍映射不一致（如券商研报 metainfo 为券商名），
     退化为纯页码匹配。

流程：
  第一步（一次性）：python scripts/eval/eval_recall.py --init-gold
    生成金标准标注模板 data/eval/gold_pages.json（预填公司名与对应 PDF 文件名），
    人工翻 PDF 把答案所在页码填入 gold 字段；
  第二步：python scripts/eval/eval_recall.py
    双臂逐题检索，计算 hit@1/3/5，输出 data/eval/recall_report.json 与控制台对比表。

用法（项目根目录、no1 环境下执行，需已配置 .env 与建库）：
    python scripts/eval/eval_recall.py --init-gold        # 生成标注模板
    python scripts/eval/eval_recall.py                    # 双臂对比（默认）
    python scripts/eval/eval_recall.py --arms vector      # 仅单路向量臂
    python scripts/eval/eval_recall.py --limit 5 --top-n 10

依赖：项目运行环境（与 run_benchmark 的检索预取相同，无需拉起 API 服务）。
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import pandas as pd

# 定位项目根目录与脚本目录并加入 sys.path，保证 import src.* 生效
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import Pipeline, RunConfig  # noqa: E402

# 数据与输出路径（与 run_benchmark 保持一致）
DATA_ROOT = PROJECT_ROOT / "data" / "stock_data"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
SUBSET_PATH = DATA_ROOT / "subset.csv"
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
GOLD_PATH = EVAL_DIR / "gold_pages.json"
REPORT_PATH = EVAL_DIR / "recall_report.json"

# 父文档目录（chunked_reports）：用于建立 file_name -> company_name 归属映射
DOCUMENTS_DIR = DATA_ROOT / "databases" / "chunked_reports"

# 评测的 k 值列表（README 口径为 Top-5，附带 1/3 便于观察排序质量）
KS = [1, 3, 5]

# 臂定义说明（仅 llm_reranking 不同，其余字段保证两臂可比）：
#   vector  单路向量检索（FAISS IndexFlatIP，top_n 直接取最近邻）
#   hybrid  向量取 30 候选 -> LLMReranker 重排 -> 取 top_n（/chat 的重排开关路径）
_ARM_NOTE = {
    "vector": "单路向量检索（llm_reranking=False）",
    "hybrid": "向量候选 + LLM 重排（llm_reranking=True，HybridRetriever）",
}


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #

def _load_dataset(limit: int) -> list[dict]:
    """读取评测集（默认 data/eval/eval_dataset.json），limit > 0 时截取前 N 题。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"评测集不存在: {DATASET_PATH}，请先运行 build_eval_dataset.py 生成。")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    if limit and len(records) > limit:
        records = records[:limit]
    print(f"[recall] 评测集共 {len(records)} 题（来源: {DATASET_PATH}）")
    return records


def _load_gold() -> dict[int, dict]:
    """读取金标准页码标注，返回 {题目 id: {"公司": [页码], ...}}。

    支持两种标注格式：
      1. 标准格式：{"id": 1, "gold": {"比亚迪": [57, 58]}}（多公司题必须用这种）
      2. 简化格式：{"id": 1, "pages": [57, 58]}（单公司题等价于 {"_single_": [57, 58]}，
         匹配时退化为纯页码匹配，不校验公司归属）
    """
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"金标准文件不存在: {GOLD_PATH}\n"
            f"请先执行: python scripts/eval/eval_recall.py --init-gold 生成模板，"
            f"人工标注页码后再运行评测。"
        )
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    gold: dict[int, dict] = {}
    for entry in raw:
        qid = entry.get("id")
        if qid is None:
            continue
        # 简化格式：无 gold 字段但有 pages 列表，归一化为单公司口径
        if not entry.get("gold") and entry.get("pages"):
            gold[int(qid)] = {"_single_": [int(p) for p in entry["pages"]]}
        elif entry.get("gold"):
            # 统一转 {公司: [int 页码]}，过滤空页码列表
            gold[int(qid)] = {
                str(c): [int(p) for p in pages if p is not None]
                for c, pages in entry["gold"].items() if pages
            }
    print(f"[recall] 金标准加载：{len(gold)} 题已标注页码")
    return gold


def _build_arm_configs(top_n: int) -> dict[str, RunConfig]:
    """构造各检索臂的 RunConfig。

    两臂除 llm_reranking 外完全一致（含 parent_document_retrieval=True 返回父页面、
    parallel_requests=1 避免 AGICTO 限流），保证召回差异只来自检索方式本身。
    """
    # 公共基底：与 /chat 服务链路同源的单题检索配置
    base = RunConfig(
        parent_document_retrieval=True,   # 返回整页并按页去重，页码口径与可溯源引用一致
        top_n_retrieval=top_n,            # 每臂取前 top_n 条排序结果（需 >= max(KS)）
        parallel_requests=1,              # 单题串行，避免 qwen-plus 重排触发限流
    )
    return {
        "vector": replace(base, llm_reranking=False),                     # 单路向量
        "hybrid": replace(base, llm_reranking=True,                       # 向量+LLM 重排
                          llm_reranking_sample_size=30),                  # 重排候选数与线上一致
    }


def _build_file_company_map() -> dict[str, str]:
    """扫描 chunked_reports 建立 file_name -> company_name 映射（结果归属判定用）。"""
    mapping: dict[str, str] = {}
    if not DOCUMENTS_DIR.exists():
        print(f"[recall] 警告：文档目录不存在 {DOCUMENTS_DIR}，公司归属将全部退化为页码匹配")
        return mapping
    for path in DOCUMENTS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            meta = doc.get("metainfo", {})
            file_name = meta.get("file_name", "")
            company = meta.get("company_name", "")
            if file_name and company:
                mapping[file_name] = company
        except Exception as e:
            print(f"[recall] 警告：读取 {path.name} 失败（跳过）: {e}")
    print(f"[recall] 文档归属映射：{len(mapping)} 份 chunked 报告")
    return mapping


# --------------------------------------------------------------------------- #
# 金标准模板生成（--init-gold）
# --------------------------------------------------------------------------- #

def _match_companies(question: str, subset_rows: list[dict]) -> list[str]:
    """从问题文本中提取公司名（复刻 _extract_companies_from_subset 的包含匹配逻辑：
    公司名按长度降序尝试，命中后从文本中移除避免重复计数）。"""
    names = sorted({(r.get("company_name") or "").strip() for r in subset_rows
                    if (r.get("company_name") or "").strip()}, key=len, reverse=True)
    found: list[str] = []
    text = question
    for name in names:
        if name in text:
            found.append(name)
            text = text.replace(name, "")
    return found


def init_gold_template(limit: int) -> None:
    """生成金标准页码标注模板：预填题目、路由到的公司名与对应 PDF 文件名，
    gold 字段留空待人工翻 PDF 标注答案所在页码。"""
    records = _load_dataset(limit)
    # 读取 subset.csv 建立公司名 -> PDF 文件名映射（标注时便于定位 PDF）
    if not SUBSET_PATH.exists():
        raise FileNotFoundError(f"subset.csv 不存在: {SUBSET_PATH}，无法预填公司信息。")
    try:
        df = pd.read_csv(SUBSET_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(SUBSET_PATH, encoding="gbk")
    subset_rows = df.to_dict(orient="records")
    company_files: dict[str, list[str]] = {}
    for r in subset_rows:
        c = (r.get("company_name") or "").strip()
        fn = (r.get("file_name") or "").strip()
        if c and fn:
            company_files.setdefault(c, []).append(fn)

    template = []
    for rec in records:
        companies = _match_companies(rec.get("question", ""), subset_rows)
        template.append({
            "id": rec["id"],
            "question": rec.get("question", ""),
            "ground_truth": rec.get("ground_truth", ""),
            "companies": companies,
            "pdf_files": {c: company_files.get(c, []) for c in companies},
            # 人工标注：填入 {"公司名": [页码, ...]}，页码为 PDF 页序号（chunk 的 page 字段口径）
            "gold": {},
        })
        if not companies:
            print(f"[recall] 警告：题目 {rec['id']} 未能匹配到公司名，请人工确认")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"[recall] 金标准模板已生成: {GOLD_PATH}")
    print("[recall] 请人工翻阅对应 PDF，把每题答案依据所在页码填入 gold 字段，例如：")
    print('        "gold": {"比亚迪": [57, 58]}（多公司题各公司分别标注；'
          '单公司题也可用简化格式 "pages": [57]）')


# --------------------------------------------------------------------------- #
# 召回判定
# --------------------------------------------------------------------------- #

def _company_hits_at_k(candidates: list[dict], company: str, gold_pages: list[int],
                       file_company_map: dict[str, str]) -> dict[int, bool]:
    """判定某公司在前 k 条候选中是否命中金标准页码。

    candidates 为该题目聚合检索结果中归属于该公司的子序列（保持排序）；
    单公司金标准（company == "_single_"）时直接用全部候选做纯页码匹配，
    容忍 file_name -> company 映射不一致（如券商研报 metainfo 存的是券商名）。
    返回 {k: 是否命中}。
    """
    if company == "_single_":
        scoped = candidates
    else:
        scoped = [r for r in candidates
                  if file_company_map.get(r.get("file_name", "")) == company]
        if not scoped:
            # 多公司题中该公司无归属结果：判定为未命中（不做跨公司兜底，避免页码撞车假命中）
            return {k: False for k in KS}
    pages_ranked = [r.get("page", 0) for r in scoped]
    hits: dict[int, bool] = {}
    for k in KS:
        hits[k] = any(p in gold_pages for p in pages_ranked[:k])
    return hits


def _question_hits(results: list[dict], gold: dict[str, list[int]],
                   file_company_map: dict[str, str]) -> dict[int, bool]:
    """题目级 hit@k：金标准中每家公司均需在前 k 条各自候选中命中页码。"""
    # 各公司分别判定后取 AND（单公司题即普通 hit@k）
    per_company = [_company_hits_at_k(results, c, pages, file_company_map)
                   for c, pages in gold.items()]
    return {k: all(h[k] for h in per_company) for k in KS}


# --------------------------------------------------------------------------- #
# 主评测流程
# --------------------------------------------------------------------------- #

def run_eval(arm_names: list[str], top_n: int, limit: int) -> None:
    """逐臂逐题执行检索-only 评测，聚合 hit@k 并落盘报告。"""
    if top_n < max(KS):
        raise ValueError(f"--top-n（{top_n}）需 >= 最大 k 值（{max(KS)}）")
    records = _load_dataset(limit)
    gold = _load_gold()
    # 未标注金标准的题目：参与检索但只记录 top 页码，不参与召回统计
    pending = [r for r in records if r["id"] not in gold or not gold[r["id"]]]
    if pending:
        print(f"[recall] 提示：{len(pending)} 题未标注金标准页码，仅记录检索结果不计入召回："
              f"{[r['id'] for r in pending]}")

    arm_configs = _build_arm_configs(top_n)
    file_company_map = _build_file_company_map()

    # 逐臂构造 Pipeline（仅初始化路径，不加载索引；索引由检索器按需加载）
    per_question: list[dict] = []
    arm_stats: dict[str, dict] = {}
    for arm in arm_names:
        if arm not in arm_configs:
            raise ValueError(f"未知臂名: {arm}（可选: {list(arm_configs)}）")
        cfg = arm_configs[arm]
        pipeline = Pipeline(DATA_ROOT, run_config=cfg)
        n_hit = {k: 0 for k in KS}
        n_valid = 0
        print(f"\n[recall] ===== 臂 {arm}（{_ARM_NOTE.get(arm, '')}）开始检索，共 {len(records)} 题 =====")
        for rec in records:
            qid, question = rec["id"], rec.get("question", "")
            entry = next((e for e in per_question if e["id"] == qid), None)
            if entry is None:
                entry = {"id": qid, "question": question, "gold": gold.get(qid, {}),
                         "arms": {}}
                per_question.append(entry)
            t0 = time.time()
            try:
                # 与 /chat 同源的检索入口：公司路由 -> 检索（-> 重排），不调生成 LLM
                processor = pipeline._new_single_question_processor()
                results, _company, _is_comp, companies = processor.retrieve_question_contexts(question)
                # 记录各公司前 5 条页码（调试用），按归属分组
                top_pages: dict[str, list[int]] = {}
                for r in results[: max(KS)]:
                    c = file_company_map.get(r.get("file_name", ""), "_unknown_")
                    top_pages.setdefault(c, []).append(r.get("page", 0))
                arm_result: dict = {"top_pages": top_pages, "error": None,
                                    "companies_routed": companies or []}
                if qid in gold and gold[qid]:
                    hits = _question_hits(results, gold[qid], file_company_map)
                    arm_result.update({f"hit@{k}": hits[k] for k in KS})
                    n_valid += 1
                    for k in KS:
                        n_hit[k] += 1 if hits[k] else 0
                entry["arms"][arm] = arm_result
                print(f"[recall] [{arm}] 题 {qid} 检索完成（{time.time() - t0:.1f}s），"
                      f"top 页码: {top_pages}")
            except Exception as e:
                entry["arms"][arm] = {"top_pages": {}, "error": str(e),
                                      "companies_routed": []}
                print(f"[recall] [{arm}] 题 {qid} 检索失败: {e}")
        arm_stats[arm] = {
            "n_evaluated": n_valid,
            **{f"hit@{k}": n_hit[k] for k in KS},
            **{f"rate@{k}": (round(n_hit[k] / n_valid, 4) if n_valid else None) for k in KS},
        }
        print(f"[recall] 臂 {arm} 汇总: " + ", ".join(
            f"hit@{k}={n_hit[k]}/{n_valid}" for k in KS))

    # ------------------------------------------------------------------ #
    # 报告落盘与控制台输出
    # ------------------------------------------------------------------ #
    report = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": str(DATASET_PATH),
            "gold_file": str(GOLD_PATH),
            "top_n": top_n,
            "ks": KS,
            "arms": {a: {"note": _ARM_NOTE.get(a, ""), "config": asdict(arm_configs[a])}
                     for a in arm_names},
            "unannotated_ids": [r["id"] for r in pending],
        },
        "summary": arm_stats,
        "per_question": per_question,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[recall] 报告已写入: {REPORT_PATH}")

    # 控制台对比表
    print("\n===== Top-k 页码召回率对比 =====")
    header = f"{'臂':<10}" + "".join(f"hit@{k:<8}" for k in KS)
    print(header)
    for arm in arm_names:
        s = arm_stats[arm]
        row = f"{arm:<10}"
        for k in KS:
            rate = s.get(f"rate@{k}")
            row += f"{rate if rate is not None else 'N/A':<10}"
        print(row)
    if len(arm_names) == 2:
        a, b = arm_names
        sa, sb = arm_stats[a], arm_stats[b]
        if sa["n_evaluated"] and sb["n_evaluated"]:
            print(f"（{b} 相对 {a} 的 hit@5 变化: "
                  f"{(sb['rate@5'] or 0) - (sa['rate@5'] or 0):+.4f}）")
    # 未命中题目清单（便于 bad case 定位）
    for arm in arm_names:
        missed = [e["id"] for e in per_question
                  if e["arms"].get(arm, {}).get(f"hit@{max(KS)}") is False]
        if missed:
            print(f"[recall] 臂 {arm} 未命中 hit@{max(KS)} 的题目: {missed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="检索-only Top-k 召回率评测（不调生成 LLM）")
    parser.add_argument("--init-gold", action="store_true",
                        help="生成金标准页码标注模板 data/eval/gold_pages.json 后退出")
    parser.add_argument("--arms", default="vector,hybrid",
                        help="参与的检索臂，逗号分隔（默认 vector,hybrid）")
    parser.add_argument("--top-n", type=int, default=10,
                        help="每臂取前 N 条排序结果（默认 10，需 >= 5）")
    parser.add_argument("--limit", type=int, default=0,
                        help="只评测前 N 题（0 表示全部）")
    args = parser.parse_args()

    if args.init_gold:
        init_gold_template(args.limit)
        return
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    if not arm_names:
        raise ValueError("--arms 不能为空")
    run_eval(arm_names, args.top_n, args.limit)


if __name__ == "__main__":
    main()
