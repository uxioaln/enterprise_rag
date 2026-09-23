# -*- coding: utf-8 -*-
"""
长程稳定性 (Long-range Stability) 自动化测试：模拟多轮对话，逐轮提问并评估
答案质量（RAGAS faithfulness），直到质量显著下降或出现幻觉为止，输出"稳定轮数"。

测试原理：
    维护对话历史 history 逐轮累积（格式 [{question, final_answer}]，与 /chat 链路
    的会话存储一致），每轮调用 Pipeline.answer_with_contexts(question, history=history)
    生成答案；待测配置内部的 L2 历史压缩（history_keep_rounds）会随轮数推进把早期
    轮摘要化，这正是本脚本要观测的长程衰减来源；每轮再用 RAGAS faithfulness
    （判官模型 gpt-4o-mini，复用 run_ragas 的 evaluator 配置）评估该轮答案质量。

用法（在项目根目录执行，需已配置 .env 中的 AGICTO_API_KEY，且对应配置已完成建库）：
    python scripts/eval/long_range_stability.py --config ours --mode sequence --max-rounds 20

输入：
    data/eval/eval_dataset.json   评估数据集（build_eval_dataset 产物）
    data/stock_data/databases*    对应配置的向量库（batch_generate 前置条件相同）

输出：
    data/eval/long_range_stability_{config}.json   逐轮明细与汇总
    data/eval/long_range_stability_{config}.md     可读报告（逐轮曲线表 + 终止轮现场）

提问模式：
    sequence  按数据集顺序逐题提问，问完一轮后循环（贴近真实长程对话，默认）
    repeat    固定数据集第一题反复追问（检验同题追问下的稳定性）

终止条件（均可通过参数调整）：
    1. 绝对幻觉：某轮 faithfulness < --abs-threshold（默认 0.6，与 analyze_badcase 阈值一致）
    2. 显著下降：某轮 faithfulness 较首个有效轮下降 >= --rel-drop（默认 0.2）
    3. 安全上限：达到 --max-rounds（默认 20）仍未触发下降，视为全程稳定

成本控制：
    --eval-every 隔轮评估（默认 1 即每轮都评；设为 2 可将判官 LLM 调用减半，
    代价是终止判定最多延后 eval_every-1 轮生效）。

依赖：项目运行环境（src.pipeline、ragas、langchain_openai），判官 LLM 走 AGICTO 平台。
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 定位项目根目录与脚本目录并加入 sys.path：
# 前者保证 import src.*，后者保证复用同目录 run_ragas 的 _init_evaluators
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.pipeline import Pipeline, configs  # noqa: E402
from run_ragas import _init_evaluators  # noqa: E402

# 数据集根目录与评估输出目录（与 batch_generate 保持一致）
DATA_ROOT = PROJECT_ROOT / "data" / "stock_data"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DATASET_PATH = EVAL_DIR / "eval_dataset.json"


def _load_dataset() -> list[dict]:
    """读取评估数据集，若不存在则提示先运行 build_eval_dataset。"""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"未找到评估数据集 {DATASET_PATH}，请先运行: python scripts/eval/build_eval_dataset.py"
        )
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_question(dataset: list[dict], round_no: int, mode: str) -> dict:
    """按提问模式选取第 round_no 轮（1-based）的问题。

    sequence：按数据集顺序循环取题；repeat：固定第一题反复追问。
    """
    if mode == "repeat":
        return dataset[0]
    return dataset[(round_no - 1) % len(dataset)]


def _eval_faithfulness_batch(samples: list, llm, embeddings) -> list[float | None]:
    """对一批样本计算 RAGAS faithfulness，返回逐样本分数（失败/NaN 记为 None）。

    与 run_ragas._run_metrics 保持同一调用方式：
    raise_exceptions=False 保证单样本失败不中断整体，NaN 统一转 None。
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.metrics import faithfulness

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness],
        llm=llm,
        embeddings=embeddings,
        show_progress=False,
        raise_exceptions=False,
    )
    # 逐行提取 faithfulness 分数（NaN -> None）
    df = result.to_pandas()
    scores: list[float | None] = []
    for i in range(len(samples)):
        val = df.iloc[i].get("faithfulness")
        try:
            val = float(val)
            if val != val:  # NaN 判定
                val = None
        except (TypeError, ValueError):
            val = None
        scores.append(val)
    return scores


def _run_session(pipeline: Pipeline, dataset: list[dict], llm, embeddings, args) -> dict:
    """执行一次多轮对话会话，返回完整测试结果（汇总 + 逐轮明细）。"""
    from ragas.dataset_schema import SingleTurnSample

    history: list[dict] = []          # 逐轮累积的对话历史（[{question, final_answer}]）
    rounds_data: list[dict] = []      # 逐轮观测记录（问题/答案/faithfulness/token）
    pending: list[dict] = []          # 待评估轮次缓冲（配合 eval_every 攒批降成本）
    first_score: float | None = None  # 首个有效 faithfulness，作为相对下降的基线
    stop_reason: str | None = None    # 终止原因（None 表示未触发终止条件）
    stop_round: int | None = None     # 触发终止的轮次

    for round_no in range(1, args.max_rounds + 1):
        # ---------- 1) 选取本轮问题 ----------
        item = _pick_question(dataset, round_no, args.mode)
        question = item.get("question", "")
        kind = item.get("kind", "string")
        print(f"\n[stability] 第 {round_no}/{args.max_rounds} 轮：{question[:50]}")

        # ---------- 2) 带历史提问，生成本轮答案 ----------
        rec: dict = {
            "round": round_no,
            "question_id": item.get("id"),
            "question": question,
            "answer": "",
            "faithfulness": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "context_tokens": None,
        }
        try:
            t0 = time.time()
            answer_dict, contexts = pipeline.answer_with_contexts(
                question, kind=kind, history=history
            )
            answer = str(answer_dict.get("final_answer", ""))
            rec["answer"] = answer
            # 落盘本轮 token 用量（prompt_tokens 为服务端真实值，context_tokens 为估算兜底）
            rec["prompt_tokens"] = answer_dict.get("prompt_tokens")
            rec["completion_tokens"] = answer_dict.get("completion_tokens")
            rec["context_tokens"] = answer_dict.get("context_tokens")
            rec["elapsed_seconds"] = round(time.time() - t0, 1)
        except Exception as err:
            # 推理异常：记录错误并终止会话（带病继续会污染长程观测语义）
            rec["error"] = f"{type(err).__name__}: {err}"
            rounds_data.append(rec)
            stop_reason = f"第 {round_no} 轮推理异常中断: {err}"
            stop_round = round_no
            break

        # 推理成功：本轮问答并入历史，供后续轮次作为多轮上下文
        history.append({"question": question, "final_answer": answer})
        rounds_data.append(rec)

        # ---------- 3) 构造本轮 RAGAS 样本，按 eval_every 攒批评估 ----------
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=[str(c) for c in (contexts or []) if c],
            reference=str(item.get("ground_truth", "") or ""),
        )
        pending.append({"rec": rec, "sample": sample})

        # 是否到达评估时点：整除 eval_every，或已是最后一轮（保证尾部轮次不漏评）
        due = (round_no % args.eval_every == 0) or (round_no == args.max_rounds)
        if not due:
            continue

        # 批量评估待评轮次（判官 LLM 调用）
        try:
            scores = _eval_faithfulness_batch(
                [p["sample"] for p in pending], llm, embeddings
            )
        except Exception as err:
            # 判官调用整体失败：本批分数记 None，不中断会话
            print(f"[stability] 第 {round_no} 轮批次 RAGAS 评估失败: {err}")
            scores = [None] * len(pending)
        newly_scored = [p["rec"] for p in pending]
        for p, s in zip(pending, scores):
            p["rec"]["faithfulness"] = s
        pending.clear()
        for r in newly_scored:
            fth = r["faithfulness"]
            print(f"[stability]   第 {r['round']} 轮 faithfulness: {fth if fth is not None else 'N/A'}")

        # ---------- 4) 终止判定：按轮次顺序检查新评分轮次 ----------
        for r in newly_scored:
            s = r["faithfulness"]
            if s is None:
                continue  # 该轮评分失败，不参与终止判定
            # 绝对幻觉：低于阈值即终止（判定优先级高于相对下降）
            if s < args.abs_threshold:
                stop_reason = (
                    f"绝对幻觉：第 {r['round']} 轮 faithfulness={s:.4f} < {args.abs_threshold}"
                )
                stop_round = r["round"]
                break
            # 记录首个有效分作为相对下降基线（首轮或首批有效轮）
            if first_score is None:
                first_score = s
                continue
            # 显著下降：较基线跌落超过 rel_drop
            if first_score - s >= args.rel_drop:
                stop_reason = (
                    f"显著下降：第 {r['round']} 轮 faithfulness={s:.4f} "
                    f"较基线 {first_score:.4f} 下降 {first_score - s:.4f} >= {args.rel_drop}"
                )
                stop_round = r["round"]
                break
        if stop_reason:
            break

    # 未触发任何终止条件：视为全程稳定
    if stop_reason is None:
        stop_reason = f"达到最大轮数 {args.max_rounds}，faithfulness 未出现显著下降或幻觉"
        stable_rounds = args.max_rounds  # 全程稳定
    else:
        stable_rounds = stop_round - 1 if stop_round is not None else 0  # 终止轮之前均为稳定轮

    # 汇总统计
    scored = [r["faithfulness"] for r in rounds_data if r["faithfulness"] is not None]
    summary = {
        "config": args.config,
        "mode": args.mode,
        "参数": {
            "max_rounds": args.max_rounds,
            "eval_every": args.eval_every,
            "abs_threshold": args.abs_threshold,
            "rel_drop": args.rel_drop,
        },
        "总轮数": len(rounds_data),
        "稳定轮数": stable_rounds,
        "终止原因": stop_reason,
        "终止轮次": stop_round,
        "首轮有效faithfulness": round(first_score, 4) if first_score is not None else None,
        "末轮有效faithfulness": round(scored[-1], 4) if scored else None,
        "有效评分轮数": len(scored),
        "平均faithfulness": round(sum(scored) / len(scored), 4) if scored else None,
        "最小faithfulness": round(min(scored), 4) if scored else None,
    }
    return {"summary": summary, "rounds": rounds_data}


def _write_markdown_report(result: dict, config: str) -> Path:
    """将测试结果渲染为 Markdown 报告（逐轮曲线表 + 终止轮现场）。"""
    s = result["summary"]
    p = s["参数"]
    lines = [
        "# 长程稳定性测试报告",
        "",
        f"- 配置：`{config}` | 模式：{s['mode']} | 最大轮数：{p['max_rounds']} | 评估间隔：每 {p['eval_every']} 轮",
        f"- 终止阈值：faithfulness < {p['abs_threshold']}（幻觉）或较基线下降 >= {p['rel_drop']}（显著下降）",
        f"- 终止原因：{s['终止原因']}",
        f"- 稳定轮数：**{s['稳定轮数']}**（总轮数 {s['总轮数']}）",
        f"- 首轮有效 faithfulness：{s['首轮有效faithfulness']} | 末轮：{s['末轮有效faithfulness']} | 最小：{s['最小faithfulness']} | 平均：{s['平均faithfulness']}",
        "",
        "## 逐轮 faithfulness 曲线",
        "",
        "| 轮次 | 问题 | faithfulness | prompt_tokens | context_tokens |",
        "|---|---|---|---|---|",
    ]
    for r in result["rounds"]:
        # 表格单元格安全处理：截断并去掉竖线与换行
        q = (r.get("question") or "")[:30].replace("|", " ").replace("\n", " ")
        fth = r.get("faithfulness")
        fth_str = f"{fth:.4f}" if fth is not None else "N/A"
        err = r.get("error")
        if err:
            fth_str += f"（异常: {err[:40].replace('|', ' ')}）"
        pt = r.get("prompt_tokens")
        ct = r.get("context_tokens")
        lines.append(
            f"| {r['round']} | {q} | {fth_str} | {pt if pt is not None else 'N/A'} | {ct if ct is not None else 'N/A'} |"
        )

    # 终止轮的问题与答案摘录（便于人工核查幻觉现场）
    term_round = next((r for r in result["rounds"] if r["round"] == s["终止轮次"]), None)
    if term_round:
        lines += [
            "",
            "## 终止轮现场",
            "",
            f"- 问题：{term_round.get('question', '')}",
            f"- 答案（前 300 字）：{(term_round.get('answer') or '')[:300]}",
            f"- faithfulness：{term_round.get('faithfulness')}",
        ]
        if term_round.get("error"):
            lines.append(f"- 异常：{term_round['error']}")
    lines.append("")

    md_path = EVAL_DIR / f"long_range_stability_{config}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="长程稳定性测试：多轮对话逐轮评估 faithfulness 直至质量显著下降或出现幻觉"
    )
    parser.add_argument(
        "--config", default="ours",
        help="待测配置名，默认 ours（启用 L1/L2 压缩，最易观测长程衰减；baseline 全量保留可作对照）",
    )
    parser.add_argument(
        "--mode", choices=["sequence", "repeat"], default="sequence",
        help="提问模式：sequence 按数据集顺序循环（默认），repeat 固定第一题反复追问",
    )
    parser.add_argument("--max-rounds", type=int, default=20, help="安全轮数上限，默认 20")
    parser.add_argument(
        "--eval-every", type=int, default=1,
        help="每隔几轮评估一次 faithfulness，默认 1（每轮都评）",
    )
    parser.add_argument(
        "--abs-threshold", type=float, default=0.6,
        help="绝对幻觉阈值，低于即终止，默认 0.6（与 analyze_badcase 一致）",
    )
    parser.add_argument(
        "--rel-drop", type=float, default=0.2,
        help="较首个有效轮的显著下降幅度，达到即终止，默认 0.2",
    )
    args = parser.parse_args()

    # 1) 参数校验
    if args.config not in configs:
        raise ValueError(f"未知配置: {args.config}，可用配置: {list(configs.keys())}")
    if args.max_rounds < 1:
        raise ValueError("--max-rounds 必须 >= 1")
    if args.eval_every < 1:
        raise ValueError("--eval-every 必须 >= 1")

    # 2) 加载评估数据集
    dataset = _load_dataset()
    if not dataset:
        raise ValueError(f"评估数据集为空: {DATASET_PATH}")
    print(f"[stability] 评估数据集: {len(dataset)} 条问题 | 配置: {args.config} | 模式: {args.mode}")

    # 3) 初始化待测 Pipeline（复用 batch_generate 的构造方式，需对应配置已完成建库）
    pipeline = Pipeline(DATA_ROOT, run_config=configs[args.config])

    # 4) 初始化 RAGAS 判官（复用 run_ragas 的 evaluator 配置：gpt-4o-mini + 超时/重试）
    llm, embeddings = _init_evaluators()

    # 5) 执行多轮对话会话
    t0 = time.time()
    result = _run_session(pipeline, dataset, llm, embeddings, args)
    result["summary"]["总耗时秒"] = round(time.time() - t0, 1)

    # 6) 写出 JSON 明细与 Markdown 报告
    json_path = EVAL_DIR / f"long_range_stability_{args.config}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    md_path = _write_markdown_report(result, args.config)

    # 7) 控制台摘要输出
    print("\n===== 长程稳定性摘要 =====")
    for k, v in result["summary"].items():
        print(f"{k}: {v}")
    print(f"\n[stability] JSON 明细: {json_path}")
    print(f"[stability] Markdown 报告: {md_path}")


if __name__ == "__main__":
    main()
