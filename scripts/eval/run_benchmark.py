# -*- coding: utf-8 -*-
"""
自动化评测流水线：一键对比"优化前（baseline）"与"优化后（optimized）"的质量与效率指标。

流程（默认 --mode both，单条命令完成双模式对比）：
    1. 加载数据：读取 answers_max.json（复用 build_eval_dataset 的解析与采样逻辑）；
    2. 预取上下文：in-process 调用与 /chat 同源的检索子系统获取各题 contexts
       （L1/L2 压缩只影响送入 LLM 的内容，不改变检索结果，故两模式共用）；
    3. 启动服务：mode=both 时拉起两个本地 uvicorn 子进程（baseline 用 --port，
       optimized 用 --port+1），分别通过 KB_ 前缀环境变量固定 L1/L2 开关
       （关闭/开启）——env 优先级高于 config.json 且热重载时重新生效，开关
       状态全程稳定；两服务 stdout/stderr 分别汇入各自日志文件供 token 解析，
       轮询 /health 直到就绪；
    4. 交错执行：mode=both 时逐题交错调用两端口（q1-baseline -> q1-optimized
       -> q2-baseline -> ...），每题两模式调用紧邻，消除整段 pass 之间的
       服务端负载/时段差异（此前"整段跑完 baseline 再跑 optimized"会让
       optimized 撞上 AGICTO 高峰时段，时延对比被服务端排队污染——上一轮
       benchmark 中 Q7 单题 +478s 即为该假信号）；单模式则单服务整 pass 执行；
    5. 质量评估：调用 RAGAS 四项指标（判官 gpt-4o-mini，复用 run_ragas 的
       evaluator 配置），对两模式的 API 答案分别打分；
    6. 生成报告：data/eval/benchmark_report.json + benchmark_report.md，
       以表格对比优化前后的质量与效率数据；结束后停止服务
       （config.json 全程不被修改，无需恢复）。

模式说明：
    baseline   优化前：KB_CONTEXT_COMPRESSION__ENABLE_L1/L2 = false（全量上下文）
    optimized  优化后：KB_CONTEXT_COMPRESSION__ENABLE_L1/L2 = true（L1 裁剪 + L2 摘要）
    注：开关经 KB_ 环境变量注入服务子进程（app/config.py 的加载顺序为
    defaults < config.json < KB_ 环境变量，env 每次热重载重新应用，不受
    config.json 变化影响）；retry_loop 等其他开关保持用户 config.json
    原值不变（两模式恒定，不影响对比公平性）。多轮场景下的 L2 长程衰减
    由 long_range_stability.py 单独覆盖，本脚本每题独立会话。

用法（在项目根目录、no1 环境下执行，需已配置 .env 与建库）：
    python scripts/eval/run_benchmark.py                     # 双模式对比（默认）
    python scripts/eval/run_benchmark.py --baseline          # 仅跑优化前
    python scripts/eval/run_benchmark.py --mode optimized    # 仅跑优化后
    python scripts/eval/run_benchmark.py --limit 5 --port 8310

依赖：项目运行环境（uvicorn、requests、ragas、langchain_openai）。
"""

import argparse
import csv
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# t 分布分位数：配对差值 95% 置信区间用（scipy 为 ragas 依赖链中的既有库）
from scipy import stats as _scipy_stats

# 定位项目根目录与脚本目录并加入 sys.path：
# 前者保证 import src.*，后者保证复用同目录 build_eval_dataset / run_ragas
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR: Path = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from build_eval_dataset import _resolve_answers_file, _extract_records  # noqa: E402
from run_ragas import _init_evaluators, METRIC_NAMES  # noqa: E402
from src.pipeline import Pipeline, max_config  # noqa: E402

# 数据与输出路径（与 batch_generate 保持一致）
DATA_ROOT = PROJECT_ROOT / "data" / "stock_data"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
SUBSET_PATH = DATA_ROOT / "subset.csv"
# 双服务各自的服务日志（stdout/stderr 汇入），token 用量按日志文件独立解析
SERVER_LOG_BASE = EVAL_DIR / "benchmark_server_base.log"
SERVER_LOG_OPT = EVAL_DIR / "benchmark_server_opt.log"
REPORT_JSON = EVAL_DIR / "benchmark_report.json"
REPORT_MD = EVAL_DIR / "benchmark_report.md"

# 服务管理参数
DEFAULT_PORT = 8310            # 避开常用 8000，防止与日常运行的服务冲突
SERVER_START_TIMEOUT = 120     # 等待 /health 就绪的超时（秒）
CHAT_TIMEOUT = 600             # 单题 /chat 超时：评测链路含多步工具调用与限流等待，放宽到 600 秒

# L1/L2 开关的 KB_ 环境变量注入值（app/config.py 加载顺序 defaults < config.json < env，
# 双下划线表示嵌套：KB_CONTEXT_COMPRESSION__ENABLE_L1 -> context_compression.enable_l1）。
# baseline 服务注入"全关"、optimized 服务注入"全开"，开关状态全程稳定，
# 替代原"写 config.json + 等 10s watcher 热重载"的切换方式（切换开销从 12s 降为 0）
_ENV_L1L2_OFF = {
    "KB_CONTEXT_COMPRESSION__ENABLE_L1": "false",
    "KB_CONTEXT_COMPRESSION__ENABLE_L2": "false",
}
_ENV_L1L2_ON = {
    "KB_CONTEXT_COMPRESSION__ENABLE_L1": "true",
    "KB_CONTEXT_COMPRESSION__ENABLE_L2": "true",
}

# /chat 的问题校验正则（与 app/schemas.py 保持一致）：必须包含英文双引号包裹的公司名
_QUOTED_COMPANY_RE = re.compile(r'"[^"]{2,50}"')
# 企业名称常见后缀（长的在前，正则交替按序匹配）：
# 包裹引号时需连同后缀整段替换为库中标准短名，避免引号外残留"股份有限公司"
# 导致 LLM 跨引号拼接出全名、与库中 company_name 不匹配而报 No report found
_COMPANY_SUFFIX_RE = re.compile(r"(股份有限公司|有限责任公司|有限公司|集团)")

# 拒答类检测正则：答案或黄金答案含这些模式视为"拒答"（未披露/无法回答类元陈述）。
# 拒答类答案（如"年报中未披露X"）的陈述无法从检索上下文逐句印证，RAGAS 打分是噪声，
# 需在聚合时单独处理：答案拒答且黄金答案也拒答 = 正确拒答（从均值剔除并单独计数），
# 答案拒答但黄金答案有数据 = 错误拒答（保留在均值中，低分合理）
_REFUSAL_RE = re.compile(
    r"未披露|未提供|未提及|未公布|无法提供|无法确定|无法比较|不可得|不包含|尚未|N/A",
    re.IGNORECASE,
)


def _is_refusal(*texts: str) -> bool:
    """判断给定文本（可传多段，任一命中即算）是否属于拒答类表述。"""
    return any(_REFUSAL_RE.search(t or "") for t in texts)


# 服务日志中的 token 用量正则：
# 1) agent 链路打印的完整 ChatCompletion repr：
#    usage=CompletionUsage(completion_tokens=N, prompt_tokens=N, total_tokens=N, ...)
#    （prompt_tokens_details=/total_tokens= 等字段因 tokens 后非等号不会被误匹配）
_RE_INPUT_TOKENS = re.compile(r"prompt_tokens=(\d+)")
_RE_OUTPUT_TOKENS = re.compile(r"completion_tokens=(\d+)")
# 2) generate_answer_from_contexts 打印的 tiktoken 估算：[上下文Token统计] 本轮 final_prompt tokens=N
_RE_EST_PROMPT = re.compile(r"final_prompt tokens=(\d+)")


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #

def _load_questions(limit: int, seed: int) -> list[dict]:
    """从 answers_max.json 加载评测问题（复用 build_eval_dataset 的候选路径与解析）。

    limit > 0 且样本超限时按固定种子随机采样，与 build_eval_dataset 行为一致。
    """
    answers_file = _resolve_answers_file()
    if answers_file is None:
        raise FileNotFoundError(
            "未找到答案文件 answers_max.json（或其后备候选），请先运行 pipeline 生成。"
        )
    print(f"[benchmark] 读取答案文件: {answers_file}")
    with open(answers_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = _extract_records(raw)
    if not records:
        raise ValueError(f"未能从 {answers_file} 中提取有效问答记录，请检查文件格式。")
    if limit and len(records) > limit:
        total = len(records)
        random.seed(seed)
        records = random.sample(records, limit)
        records = [{**r, "id": i + 1} for i, r in enumerate(records)]
        print(f"[benchmark] 样本数 {total} 超过上限，随机抽取 {limit} 条（seed={seed}）")
    print(f"[benchmark] 评测集共 {len(records)} 题")
    return records


def _load_company_names() -> list[str]:
    """从 subset.csv 读取公司名列，用于给问题补英文双引号（/chat 校验要求）。"""
    if not SUBSET_PATH.exists():
        return []
    with open(SUBSET_PATH, "r", encoding="utf-8") as f:
        return [row["company_name"].strip() for row in csv.DictReader(f)
                if (row.get("company_name") or "").strip()]


def _ensure_quoted_company(question: str, company_names: list[str]) -> str:
    """确保问题包含英文双引号包裹的公司名（满足 /chat 接口校验）。

    已含引号则原样返回；否则将问题中"公司名+紧跟的企业后缀"整段替换为
    引号包裹的库中标准短名（如"比亚迪股份有限公司"替换为 "比亚迪"），
    保证 LLM 从引号内提取的公司名与库中 company_name 精确一致；
    同位置命中多个公司名时取更长者（更精确）；找不到公司名时原样返回
    （该题会因校验失败被记为 error）。
    """
    if _QUOTED_COMPANY_RE.search(question):
        return question
    best: tuple[int, int, str] | None = None  # (起始位置, 结束位置, 标准短名)
    for c in company_names:
        if not c:
            continue
        idx = question.find(c)
        if idx < 0:
            continue
        # 公司名后紧跟的企业后缀（如有）一并纳入替换范围
        m = _COMPANY_SUFFIX_RE.match(question, idx + len(c))
        end = m.end() if m else idx + len(c)
        # 取出现位置最靠前的；同位置取跨度更长的
        if best is None or idx < best[0] or (idx == best[0] and end > best[1]):
            best = (idx, end, c)
    if best is None:
        return question
    idx, end, company = best
    return question[:idx] + f'"{company}"' + question[end:]


# --------------------------------------------------------------------------- #
# 服务生命周期与在线开关
# --------------------------------------------------------------------------- #

def _start_server(port: int, log_path: Path,
                  env_extra: dict[str, str] | None = None) -> tuple[subprocess.Popen, object]:
    """启动本地 API 服务子进程（uvicorn），stdout/stderr 汇入指定日志文件供 token 解析。

    env_extra：注入子进程的 KB_ 前缀环境变量（固定 L1/L2 开关）；
    env 优先级高于 config.json 且每次热重载重新应用，开关状态全程稳定。
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"      # 关闭子进程输出缓冲，保证 token 日志及时落盘
    env["PYTHONIOENCODING"] = "utf-8"  # 固定子进程输出编码，避免中文日志乱码
    if env_extra:
        env.update(env_extra)          # 注入 L1/L2 开关等 KB_ 配置覆盖
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(PROJECT_ROOT), stdout=log_f, stderr=subprocess.STDOUT, env=env,
    )
    print(f"[benchmark] 启动 API 服务: port={port}（env 覆盖: {env_extra or '无'}），服务日志: {log_path}")
    api_base = f"http://127.0.0.1:{port}"
    deadline = time.time() + SERVER_START_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            log_f.close()
            raise RuntimeError(f"API 服务启动失败（退出码 {proc.returncode}），详见 {log_path}")
        try:
            if requests.get(f"{api_base}/health", timeout=3).status_code == 200:
                print("[benchmark] API 服务就绪")
                return proc, log_f
        except requests.RequestException:
            pass
        time.sleep(2)
    proc.terminate()
    log_f.close()
    raise RuntimeError(f"等待 API 服务就绪超时（{SERVER_START_TIMEOUT} 秒），详见 {log_path}")


def _stop_server(proc: subprocess.Popen) -> None:
    """停止 API 服务子进程：先 terminate，超时则 kill。"""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------- #
# token 日志解析
# --------------------------------------------------------------------------- #

def _parse_new_tokens(log_path: Path, lines_parsed: int) -> tuple[dict, int]:
    """解析服务日志自第 lines_parsed 行之后的新增行，累计 token 用量。

    返回 (token 统计, 新的已完成行数)。
    末行若无换行符视为写入中，留待下次解析，避免截断行漏配/误配。
    """
    stats = {"input_tokens": 0, "completion_tokens": 0,
             "llm_calls": 0, "final_prompt_tokens_est": 0}
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return stats, lines_parsed
    # split 后最后一段是空串（以换行结尾）或写入中的半行，统一剔除
    complete_lines = content.split("\n")[:-1]
    for line in complete_lines[lines_parsed:]:
        m = _RE_INPUT_TOKENS.search(line)
        if m:
            stats["input_tokens"] += int(m.group(1))
            stats["llm_calls"] += 1
        m = _RE_OUTPUT_TOKENS.search(line)
        if m:
            stats["completion_tokens"] += int(m.group(1))
        m = _RE_EST_PROMPT.search(line)
        if m:
            stats["final_prompt_tokens_est"] += int(m.group(1))
    return stats, len(complete_lines)


# --------------------------------------------------------------------------- #
# 评测执行
# --------------------------------------------------------------------------- #

def _ask_one(mode_name: str, api_base: str, q: dict, company_names: list[str],
             log_path: Path, lines_parsed: int) -> tuple[dict, int]:
    """单题执行：调 /chat（stream=false）取答案，并解析该题新增的 token 用量。

    返回 (本题记录, 新的日志解析行数游标)。log_path 为该模式所属服务的日志
    文件（双服务各自独立，游标互不干扰）；session_id 带模式前缀，两模式
    各自独立会话，历史互不污染。
    """
    prefix = "bench-base" if mode_name == "baseline" else "bench-opt"
    # 问题规范化：补英文双引号公司名以满足 /chat 校验
    question_sent = _ensure_quoted_company(q["question"], company_names)
    rec: dict = {
        "id": q["id"],
        "question": q["question"],
        "question_sent": question_sent,
        "ground_truth": q.get("ground_truth", ""),
        "answer": "",
        "answer_statement": "",
        "elapsed_seconds": None,
        "input_tokens": 0,
        "completion_tokens": 0,
        "llm_calls": 0,
        "final_prompt_tokens_est": 0,
    }
    try:
        resp = requests.post(
            f"{api_base}/chat",
            json={"session_id": f"{prefix}-q{q['id']:03d}", "question": question_sent},
            params={"stream": "false"},
            timeout=CHAT_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            rec["answer"] = str(data.get("answer", ""))
            # 完整陈述句版答案：RAGAS faithfulness 依赖 response 可拆解为陈述句，
            # 短答案（人名/单个数字）拆不出 claim 会导致判官返回 NaN（记 null）
            rec["answer_statement"] = str(data.get("answer_statement", ""))
            rec["elapsed_seconds"] = data.get("elapsed_seconds")
        else:
            rec["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
    except requests.RequestException as err:
        rec["error"] = f"{type(err).__name__}: {err}"

    # 解析本题期间该服务日志新增的 token 用量
    stats, lines_parsed = _parse_new_tokens(log_path, lines_parsed)
    rec.update(stats)
    status = "成功" if not rec.get("error") else f"失败({rec['error'][:60]})"
    print(f"[benchmark] [{mode_name}] 问题 {q['id']}: {status} | input_tokens={rec['input_tokens']} "
          f"| 用时={rec['elapsed_seconds']}s")
    return rec, lines_parsed


def _run_pass(mode_name: str, questions: list[dict], company_names: list[str],
              port: int, log_path: Path) -> list[dict]:
    """单模式整 pass 执行：逐题调 /chat 取答案（--mode baseline / optimized 使用）。

    pass 开始前先推进一次日志游标，跳过服务启动期间已存在的日志行，
    避免启动流量被错误归入第 1 题。
    """
    api_base = f"http://127.0.0.1:{port}"
    mode_label = "基准（优化前，L1/L2 关闭）" if mode_name == "baseline" else "优化（优化后，L1/L2 开启）"
    print(f"\n[benchmark] ===== 开始{mode_label}评测：共 {len(questions)} 题 =====")
    # 跳过本 pass 首题之前日志中已存在的行（如服务启动流量），仅推进游标
    _, lines_parsed = _parse_new_tokens(log_path, 0)
    records: list[dict] = []
    t_pass = time.time()
    for q in questions:
        rec, lines_parsed = _ask_one(mode_name, api_base, q, company_names, log_path, lines_parsed)
        records.append(rec)
    print(f"[benchmark] {mode_label}评测完成，总耗时 {time.time() - t_pass:.1f} 秒")
    return records


def _run_interleaved(questions: list[dict], company_names: list[str],
                     base_port: int, opt_port: int) -> tuple[list[dict], list[dict]]:
    """双模式逐题交错执行：q1-baseline -> q1-optimized -> q2-baseline -> ...

    每题的两模式调用紧邻（间隔仅一次请求耗时），两模式经受几乎相同的
    AGICTO 服务端负载与时段条件，时延对比不再被"整段 pass 先后"的时段
    差异污染（上一轮 benchmark optimized 平均时延 +70.5% 中约 95% 来自
    Q7/Q9 两题撞上服务端排队，属假信号）。返回 (baseline 记录, optimized 记录)。
    """
    base_api = f"http://127.0.0.1:{base_port}"
    opt_api = f"http://127.0.0.1:{opt_port}"
    # 跳过两服务启动期间的日志行（如有），仅推进游标
    _, cursor_base = _parse_new_tokens(SERVER_LOG_BASE, 0)
    _, cursor_opt = _parse_new_tokens(SERVER_LOG_OPT, 0)
    base_records: list[dict] = []
    opt_records: list[dict] = []
    t0 = time.time()
    print(f"\n[benchmark] ===== 开始交错评测：共 {len(questions)} 题，每题先 baseline 后 optimized =====")
    for q in questions:
        # 同一题先跑 baseline 再跑 optimized，两次调用紧邻以消除时段差异
        rec_b, cursor_base = _ask_one("baseline", base_api, q, company_names,
                                      SERVER_LOG_BASE, cursor_base)
        base_records.append(rec_b)
        rec_o, cursor_opt = _ask_one("optimized", opt_api, q, company_names,
                                     SERVER_LOG_OPT, cursor_opt)
        opt_records.append(rec_o)
    print(f"[benchmark] 交错评测完成，总耗时 {time.time() - t0:.1f} 秒")
    return base_records, opt_records


def _eval_ragas(records: list[dict], contexts_map: dict, llm, embeddings,
                runs: int = 1) -> None:
    """对一批 API 答案计算 RAGAS 四项指标（可多轮取均值），回填 records 的 scores 字段。

    调用方式与 run_ragas._run_metrics 一致（raise_exceptions=False，NaN 转 None）；
    retrieved_contexts 使用预取的同源检索结果，reference 使用黄金答案。

    response 口径统一：优先用完整陈述句 answer_statement（短答案拆不出 claim
    会被判官记 NaN），回落 final_answer。所有四项指标共用同一 response，
    不再按指标拆分——拆分口径曾导致 answer_relevancy 从 ~0.75 暴跌至 ~0.43
    （完整 answer 过长时 RAGAS 生成候选问题语义发散，嵌入相似度失真）。

    runs > 1 时同一批样本重复评估 runs 轮，逐题逐指标取非 null 值的均值：
    gpt-4o-mini 判官在 temperature=0 下仍存在批次级随机性（10 题样本中，
    输入完全相同的 context_precision 单题翻转 0.867 -> 0.500 即为该噪声）。
    """
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    # 构建样本：所有指标共用同一 response（answer_statement 优先，回落 final_answer）
    samples: list = []
    valid_idx: list[int] = []
    for i, rec in enumerate(records):
        # RAGAS response 口径：优先用完整陈述句（answer_statement），
        # 回落 final_answer；短答案（人名/数字）拆不出陈述句会让判官返回 NaN
        answer = (rec.get("answer_statement") or "").strip() or (rec.get("answer") or "").strip()
        contexts = contexts_map.get(rec.get("id")) or []
        if not answer or not contexts:
            continue  # 无答案或无上下文无法参与指标计算
        samples.append(SingleTurnSample(
            user_input=rec.get("question", ""),
            response=answer,
            retrieved_contexts=[str(c) for c in contexts if c],
            reference=str(rec.get("ground_truth", "") or ""),
        ))
        valid_idx.append(i)
    print(f"[benchmark] RAGAS 评估：{len(valid_idx)}/{len(records)} 条有效样本参与计算"
          f"（每指标 {runs} 轮取均值）")
    if not valid_idx:
        return

    all_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    # 逐轮评估并按（题目记录下标, 指标）收集各轮分值，最后取均值
    collected: dict[int, dict[str, list[float]]] = {
        i: {m: [] for m in METRIC_NAMES} for i in valid_idx
    }

    def _run_once() -> None:
        """单轮评估：解析逐题分值，非 NaN 的追加进 collected。"""
        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=all_metrics,
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        for offset, rec_idx in enumerate(valid_idx):
            row = df.iloc[offset]
            for metric in all_metrics:
                val = row.get(metric.name)
                try:
                    val = float(val)
                    if val != val:  # NaN 判定
                        val = None
                except (TypeError, ValueError):
                    val = None
                if val is not None:
                    collected[rec_idx][metric.name].append(val)

    for run_no in range(1, runs + 1):
        if runs > 1:
            print(f"[benchmark] RAGAS 第 {run_no}/{runs} 轮...")
        _run_once()

    # 逐题逐指标取多轮均值（全部轮次均为 null 则记 None）
    for rec_idx in valid_idx:
        scores: dict = {}
        for m in METRIC_NAMES:
            vals = collected[rec_idx][m]
            scores[m] = round(statistics.mean(vals), 4) if vals else None
        records[rec_idx]["scores"] = scores


# --------------------------------------------------------------------------- #
# 汇总与报告
# --------------------------------------------------------------------------- #

# 效率指标聚合键：记录字段 -> 报告指标名
_EFFICIENCY_KEYS = [
    ("input_tokens", "平均每题input_tokens_服务端"),
    ("completion_tokens", "平均每题completion_tokens_服务端"),
    ("llm_calls", "平均每题LLM调用次数"),
    ("final_prompt_tokens_est", "平均每题final_prompt_tokens_估算"),
    ("elapsed_seconds", "平均响应时延_秒"),
]


def _aggregate_pass(mode_name: str, records: list[dict]) -> dict:
    """汇总单模式的指标：RAGAS 四项质量均值 + token/时延效率均值。

    质量口径补充（本次修订）：
    - 每项 RAGAS 指标附带"有效题数/null题数"（判官对短答案拆不出陈述句时返回
      NaN 记 null，null 不计入均值，避免分母虚高/虚低误导）；
    - 拒答类单独处理：答案拒答且黄金答案也拒答 = 正确拒答（从 RAGAS 均值剔除，
      "未披露X"类元陈述无法从上下文逐句印证，打分是噪声）；
      答案拒答但黄金答案有数据 = 错误拒答（保留在均值中，低分合理）。
    """
    agg: dict = {"config": mode_name, "总题数": len(records)}
    ok = [r for r in records if not r.get("error")]
    agg["有效题数"] = len(ok)
    agg["失败题数"] = len(records) - len(ok)

    # 拒答分类：答案（陈述句或 final_answer 任一）含拒答模式即视为拒答
    correct_refusal_ids: set = set()
    wrong_refusal = 0
    for r in ok:
        ans_text = (r.get("answer_statement") or "") + (r.get("answer") or "")
        if _is_refusal(ans_text):
            if _is_refusal(r.get("ground_truth") or ""):
                correct_refusal_ids.add(r.get("id"))
            else:
                wrong_refusal += 1
    agg["正确拒答数"] = len(correct_refusal_ids)
    agg["错误拒答数"] = wrong_refusal

    # 质量指标均值（忽略 None；剔除正确拒答记录）
    for m in METRIC_NAMES:
        vals: list[float] = []
        null_cnt = 0
        for r in records:
            if r.get("id") in correct_refusal_ids:
                continue  # 正确拒答：不计入 RAGAS 均值
            scores = r.get("scores")
            if scores is None:
                continue  # 未参评（请求失败/无答案/无上下文），不计 null
            v = scores.get(m)
            if v is None:
                null_cnt += 1  # 判官返回 NaN 的参评记录
            else:
                vals.append(v)
        agg[m] = round(statistics.mean(vals), 4) if vals else None
        agg[f"{m}_有效题数"] = len(vals)
        agg[f"{m}_null题数"] = null_cnt

    # 效率指标均值（基于成功题）
    for field, label in _EFFICIENCY_KEYS:
        vals = [r.get(field) for r in ok if r.get(field) is not None]
        agg[label] = round(statistics.mean(vals), 1) if vals else None
    return agg


def _build_comparison(base: dict, opt: dict,
                      base_records: list[dict], opt_records: list[dict]) -> dict:
    """双模式对比：质量差值（优化后-优化前）与效率变化率。

    质量差值附逐题配对统计（降方差方案 A）：按题 id 对齐两臂记录，剔除任一臂的
    正确拒答题（与均值口径一致），对两臂均有非 null 分值的题计算配对差值
    （optimized - baseline）的均值与 95% 置信区间（t 分布，配对数 >= 2 才计算）；
    置信区间不含 0 视为统计显著。均值口径的 diff 保留原样（不变）。
    """
    comp: dict = {"质量指标（diff=优化后-优化前，正值更优）": {}}
    # 配对准备：按题 id 对齐两臂记录，剔除正确拒答题（两臂取并集）
    base_by_id = {r.get("id"): r for r in base_records}
    opt_by_id = {r.get("id"): r for r in opt_records}
    excluded = set(base.get("正确拒答题id") or []) | set(opt.get("正确拒答题id") or [])
    paired_ids = [qid for qid in base_by_id
                  if qid in opt_by_id and qid not in excluded]
    for m in METRIC_NAMES:
        b, o = base.get(m), opt.get(m)
        entry = {
            "baseline": b, "optimized": o,
            "diff": round(o - b, 4) if (b is not None and o is not None) else None,
        }
        # 逐题配对差值：两臂该指标均为非 null 才纳入
        diffs = [
            opt_by_id[qid]["scores"][m] - base_by_id[qid]["scores"][m]
            for qid in paired_ids
            if (base_by_id[qid].get("scores") and opt_by_id[qid].get("scores")
                and base_by_id[qid]["scores"].get(m) is not None
                and opt_by_id[qid]["scores"].get(m) is not None)
        ]
        if len(diffs) >= 2:
            mean_d = statistics.mean(diffs)
            se = statistics.stdev(diffs) / (len(diffs) ** 0.5)
            # t 分布 97.5% 分位数（双侧 95% CI，自由度 n-1）
            t_crit = float(_scipy_stats.t.ppf(0.975, len(diffs) - 1))
            ci_low, ci_high = mean_d - t_crit * se, mean_d + t_crit * se
            entry["配对数"] = len(diffs)
            entry["配对差值均值"] = round(mean_d, 4)
            entry["差值95%CI"] = [round(ci_low, 4), round(ci_high, 4)]
            entry["统计显著(CI不含0)"] = bool(ci_low * ci_high > 0)
        comp["质量指标（diff=优化后-优化前，正值更优）"][m] = entry
    eff: dict = {}
    for _, label in _EFFICIENCY_KEYS:
        b, o = base.get(label), opt.get(label)
        # 变化率 = (优化前-优化后)/优化前，正值表示优化后更省/更快
        rate = round((b - o) / b, 4) if (b is not None and o is not None and b != 0) else None
        eff[label] = {"baseline": b, "optimized": o, "变化率_正值更省": rate}
    comp["效率指标"] = eff
    return comp


def _fmt(v) -> str:
    """报告表格数值格式化：None 显示 N/A。"""
    return "N/A" if v is None else str(v)


def _detect_anomalies(base_records: list[dict], opt_records: list[dict]) -> list[dict]:
    """扫描双臂逐题记录，检测低分与异常退化题目，供后续逐题分析。

    检测规则：
    1. faithfulness 为 0 分（任一臂）：短答案 claim 不可验证 / 误拒答 / 判官噪声
    2. 某指标 optimized 比 baseline 下降 > 0.1：L1 压缩可能丢失关键信息
    3. answer_relevancy < 0.3：response 过短导致 RAGAS 候选问题语义发散
    4. context_recall < 0.5：检索未覆盖黄金答案所需信息
    5. 误拒答：answer 含拒答语义但 ground_truth 有实际数据（检索失败）
    每条异常包含：题号、异常类型、双臂分数、双臂答案摘要、黄金答案摘要、诊断分类。
    """
    base_by_id = {r.get("id"): r for r in base_records}
    opt_by_id = {r.get("id"): r for r in opt_records}
    anomalies: list[dict] = []
    _REFUSAL_WORDS = ("未披露", "未提供", "未包含", "未提及", "无法确定", "无法比较")

    for qid in sorted(base_by_id):
        b = base_by_id[qid]
        o = opt_by_id.get(qid, {})
        bs = b.get("scores") or {}
        os_ = o.get("scores") or {}
        tags: list[str] = []

        # 1) faithfulness 零分检测
        bf, of = bs.get("faithfulness"), os_.get("faithfulness")
        if bf == 0.0 or of == 0.0:
            tags.append("faithfulness零分")

        # 2) 指标退化检测（optimized - baseline < -0.1）
        for m in METRIC_NAMES:
            bv, ov = bs.get(m), os_.get(m)
            if bv is not None and ov is not None and ov - bv < -0.1:
                tags.append(f"{m}退化{round(ov - bv, 4)}")

        # 3) 低 answer_relevancy
        if (bs.get("answer_relevancy") or 0) < 0.3 or (os_.get("answer_relevancy") or 0) < 0.3:
            tags.append("低relevancy")

        # 4) 低 context_recall
        if (bs.get("context_recall") or 1) < 0.5 or (os_.get("context_recall") or 1) < 0.5:
            tags.append("低recall")

        # 5) 误拒答检测：answer 含拒答语义但 GT 有实际数据
        b_ans = (b.get("answer") or "")[:60]
        gt = (b.get("ground_truth") or "")[:60]
        b_refusal = any(w in b_ans for w in _REFUSAL_WORDS)
        gt_has_data = not any(w in gt for w in _REFUSAL_WORDS) and len(gt) > 10
        if b_refusal and gt_has_data:
            tags.append("误拒答(检索失败)")

        if not tags:
            continue  # 无异常的题跳过

        anomalies.append({
            "id": qid,
            "question": b.get("question", "")[:80],
            "异常类型": tags,
            "baseline_scores": bs,
            "optimized_scores": os_,
            "baseline_answer": (b.get("answer") or "")[:100],
            "optimized_answer": (o.get("answer") or "")[:100],
            "ground_truth": gt,
        })
    return anomalies


def _write_reports(report: dict) -> None:
    """写出 JSON 全量报告与 Markdown 对比报告。"""
    # 1) JSON 全量报告（含逐题明细）
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 2) Markdown 对比报告
    meta = report["meta"]
    lines = [
        "# 自动化评测报告（Benchmark：优化前 vs 优化后）",
        "",
        f"- 生成时间：{meta['生成时间']} | 模式：{meta['模式']} | 题数：{meta['题数']}"
        f" | 服务端口：{meta['端口']} | RAGAS轮数：{meta.get('ragas_runs', 1)}",
        "- 答案来源：本地 /chat 接口（stream=false，每题独立会话）；"
        "上下文：同源检索预取；RAGAS 判官：gpt-4o-mini（AGICTO，temperature=0）",
        "- 优化定义：baseline = L1/L2 上下文压缩关闭；optimized = L1/L2 开启"
        "（双服务实例 KB_ 环境变量固定开关，逐题交错执行消除时段差异）",
        "- RAGAS response 口径（统一）：优先使用完整陈述句 answer_statement（短答案拆不出"
        "陈述句会被判官记 null，null 不计入均值并单独报告有效/null 题数）；answer_relevancy"
        " 也用 answer_statement（曾试过用完整 answer 导致 relevancy 从 ~0.75 暴跌至 ~0.43，"
        "已回退）；正确拒答（答案与黄金答案均拒答）不计入 RAGAS 均值，单独统计；"
        "每指标多轮评估取逐题均值，对比表附配对差值 95% 置信区间（t 分布）",
        "",
    ]
    for arm, title in (("baseline", "优化前（baseline，L1/L2 关闭）"),
                       ("optimized", "优化后（optimized，L1/L2 开启）")):
        if arm not in report["passes"]:
            continue
        s = report["passes"][arm]["summary"]
        lines += [
            f"## {title}",
            "",
            f"- 有效题数：{s['有效题数']}/{s['总题数']}"
            + (f"（失败 {s['失败题数']} 题）" if s["失败题数"] else ""),
            f"- 拒答统计：正确拒答 {s.get('正确拒答数', 0)} 题（不计入 RAGAS 均值）"
            f" | 错误拒答 {s.get('错误拒答数', 0)} 题（计入均值）",
            "",
            "| 指标 | 数值（有效/null 题数） |",
            "|---|---|",
        ]
        for m in METRIC_NAMES:
            valid = s.get(f"{m}_有效题数")
            nulls = s.get(f"{m}_null题数")
            detail = "" if valid is None else f"（{valid}/{nulls}）"
            lines.append(f"| {m} | {_fmt(s.get(m))}{detail} |")
        for _, label in _EFFICIENCY_KEYS:
            lines.append(f"| {label} | {_fmt(s.get(label))} |")
        lines.append("")

    if "comparison" in report:
        comp = report["comparison"]
        lines += [
            "## 优化前后对比",
            "",
            "### 质量指标（RAGAS，越高越好；diff = optimized - baseline）",
            "",
            "| 指标 | 优化前 | 优化后 | 差值 | 配对差值均值 | 差值95%CI | CI不含0(显著) |",
            "|---|---|---|---|---|---|---|",
        ]
        for m, d in comp["质量指标（diff=优化后-优化前，正值更优）"].items():
            ci = d.get("差值95%CI")
            ci_txt = f"[{ci[0]}, {ci[1]}]" if ci else "N/A"
            lines.append(
                f"| {m} | {_fmt(d['baseline'])} | {_fmt(d['optimized'])} | {_fmt(d['diff'])} "
                f"| {_fmt(d.get('配对差值均值'))} | {ci_txt} | {_fmt(d.get('统计显著(CI不含0)'))} |"
            )
        lines += [
            "",
            "### 效率指标（变化率为正 = 优化后更省/更快）",
            "",
            "| 指标 | 优化前 | 优化后 | 变化率 |",
            "|---|---|---|---|",
        ]
        for label, d in comp["效率指标"].items():
            lines.append(
                f"| {label} | {_fmt(d['baseline'])} | {_fmt(d['optimized'])} | {_fmt(d['变化率_正值更省'])} |"
            )
        lines.append("")

    # 异常数据明细（低分题、指标退化题、误拒答题，供逐题分析）
    anomalies = report.get("anomalies") or []
    if anomalies:
        lines += [
            "",
            "## 异常数据明细（低分 / 退化 / 误拒答，供逐题分析）",
            "",
            f"共 {len(anomalies)} 条异常记录：",
            "",
        ]
        for a in anomalies:
            lines.append(f"### id={a['id']}：{a['question']}")
            lines.append(f"- 异常类型：{', '.join(a['异常类型'])}")
            lines.append(f"- baseline 分数：{a['baseline_scores']}")
            lines.append(f"- optimized 分数：{a['optimized_scores']}")
            lines.append(f"- baseline 答案：{a['baseline_answer']}")
            lines.append(f"- optimized 答案：{a['optimized_answer']}")
            lines.append(f"- 黄金答案：{a['ground_truth']}")
            lines.append("")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[benchmark] JSON 报告: {REPORT_JSON}")
    print(f"[benchmark] Markdown 报告: {REPORT_MD}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="自动化评测流水线：经本地 /chat 接口对比优化前后的 RAGAS 质量与 token 效率"
    )
    parser.add_argument(
        "--mode", choices=["both", "baseline", "optimized"], default="both",
        help="评测模式：both=双模式对比（默认），baseline=仅优化前，optimized=仅优化后",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="仅跑基准模式（等价于 --mode baseline）",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"本地 API 服务端口，默认 {DEFAULT_PORT}")
    parser.add_argument("--limit", type=int, default=30,
                        help="采样题数上限，0=全量，默认 30（10 题样本下单题翻转即可扰动均值，"
                             "扩到 30 题降低 RAGAS 判官随机性与离群时延的干扰）")
    parser.add_argument("--seed", type=int, default=42, help="随机采样种子，默认 42")
    parser.add_argument(
        "--ragas-runs", type=int, default=3,
        help="RAGAS 每指标重复评估轮数（逐题取均值降判官随机摆动），默认 3",
    )
    parser.add_argument(
        "--from-report", type=str, default="",
        help="从已有报告复用答案（跳过耗时的 /chat 阶段），指定报告路径；"
             "仅重跑 RAGAS 评估，适合答案已生成但 RAGAS 失败/超时的场景",
    )
    args = parser.parse_args()
    if args.baseline:
        args.mode = "baseline"  # --baseline 快捷方式：仅跑优化前

    # 1) 加载 .env（in-process 检索与 RAGAS 判官均依赖其中的 AGICTO_API_KEY）
    load_dotenv()

    # --from-report 模式：从已有报告复用答案，跳过 /chat 阶段，仅重跑 RAGAS
    if args.from_report:
        print(f"[benchmark] 从已有报告复用答案：{args.from_report}")
        with open(args.from_report, "r", encoding="utf-8") as f:
            old_report = json.load(f)
        passes = {}
        for name, data in old_report.get("passes", {}).items():
            records = data.get("records", [])
            # 清除旧 scores，RAGAS 会重新填充
            for rec in records:
                rec.pop("scores", None)
            passes[name] = records
            print(f"[benchmark] 复用 {name} 模式 {len(records)} 条答案")
        # 从复用记录中提取问题列表（用于预取上下文）
        questions = []
        for rec in passes.get("baseline") or passes.get("optimized") or []:
            questions.append({
                "id": rec.get("id"),
                "question": rec.get("question", ""),
                "question_sent": rec.get("question_sent", rec.get("question", "")),
                "ground_truth": rec.get("ground_truth", ""),
            })
        company_names = _load_company_names()
        if not questions:
            print("[benchmark] 报告中未找到有效记录，退出")
            return
    else:
        # 2) 加载评测问题与公司名（用于补双引号）
        questions = _load_questions(args.limit, args.seed)
        company_names = _load_company_names()

    # 3) 预取各题检索上下文（与 /chat 同源检索；L1/L2 不改变检索结果，两模式共用）
    print("[benchmark] 预取检索上下文（in-process，与 /chat 同一检索子系统）...")
    pipeline = Pipeline(DATA_ROOT, run_config=max_config)
    contexts_map: dict = {}
    for q in questions:
        try:
            contexts, _pages = pipeline._retrieve_contexts(q["question"])
            contexts_map[q["id"]] = [str(c) for c in contexts if c]
        except Exception as err:
            print(f"[benchmark] 问题 {q['id']} 检索预取失败: {err}")
            contexts_map[q["id"]] = []
    print(f"[benchmark] 上下文预取完成：{sum(1 for v in contexts_map.values() if v)}/{len(questions)} 题有上下文")

    if not args.from_report:
        # 4) 启动服务并执行评测：mode=both 起双服务（env 固定开关）逐题交错执行；
        #    单模式起单服务整 pass 执行。config.json 全程不被修改（无需备份恢复）。
        passes: dict[str, list[dict]] = {}
        if args.mode == "both":
            proc_base, log_base = _start_server(args.port, SERVER_LOG_BASE, _ENV_L1L2_OFF)
            try:
                proc_opt, log_opt = _start_server(args.port + 1, SERVER_LOG_OPT, _ENV_L1L2_ON)
            except Exception:
                # optimized 服务启动失败：先清理已就绪的 baseline 服务再抛出
                _stop_server(proc_base)
                log_base.close()
                raise
            try:
                print(f"[benchmark] 双服务就绪：baseline（L1/L2 关）端口 {args.port} / "
                      f"optimized（L1/L2 开）端口 {args.port + 1}")
                base_records, opt_records = _run_interleaved(
                    questions, company_names, args.port, args.port + 1)
                passes["baseline"] = base_records
                passes["optimized"] = opt_records
            finally:
                # 无论成败：停止两服务
                _stop_server(proc_opt)
                log_opt.close()
                _stop_server(proc_base)
                log_base.close()
        else:
            # 单模式：env 固定对应开关，单服务整 pass 执行
            env_extra = _ENV_L1L2_OFF if args.mode == "baseline" else _ENV_L1L2_ON
            log_path = SERVER_LOG_BASE if args.mode == "baseline" else SERVER_LOG_OPT
            proc, log_f = _start_server(args.port, log_path, env_extra)
            try:
                passes[args.mode] = _run_pass(args.mode, questions, company_names,
                                               args.port, log_path)
            finally:
                # 无论成败：停止服务
                _stop_server(proc)
                log_f.close()

    # 7) RAGAS 质量评估（服务已停止，判官独立调用 AGICTO；多轮取均值降判官方差）
    llm, embeddings = _init_evaluators()
    for name, records in passes.items():
        print(f"\n[benchmark] 对 {name} 模式的 {len(records)} 条答案计算 RAGAS 指标...")
        try:
            _eval_ragas(records, contexts_map, llm, embeddings, runs=args.ragas_runs)
        except Exception as err:
            print(f"[benchmark] {name} 模式 RAGAS 评估失败: {err}")

    # 6) 汇总与报告
    report: dict = {
        "meta": {
            "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "模式": args.mode,
            "题数": len(questions),
            "端口": args.port,
            "答案来源": "复用已有报告（--from-report）" if args.from_report else "本地 /chat 接口（stream=false）",
            "优化定义": "baseline=L1/L2关闭，optimized=L1/L2开启（双服务 KB_ 环境变量固定开关，逐题交错执行）",
        },
        "passes": {
            name: {"summary": _aggregate_pass(name, records), "records": records}
            for name, records in passes.items()
        },
    }
    if "baseline" in report["passes"] and "optimized" in report["passes"]:
        report["comparison"] = _build_comparison(
            report["passes"]["baseline"]["summary"],
            report["passes"]["optimized"]["summary"],
            passes["baseline"],
            passes["optimized"],
        )
        # 异常数据检测：低分题、指标退化题、误拒答题（供后续逐题分析）
        report["anomalies"] = _detect_anomalies(passes["baseline"], passes["optimized"])
    _write_reports(report)

    # 9) 控制台摘要
    print("\n===== Benchmark 摘要 =====")
    if report.get("anomalies"):
        print(f"异常数据：{len(report['anomalies'])} 条（低分/退化/误拒答，详见报告）")
    for name, data in report["passes"].items():
        s = data["summary"]
        print(f"\n[{name}] 有效题数 {s['有效题数']}/{s['总题数']}"
              f" | 正确拒答 {s.get('正确拒答数', 0)}（剔除）| 错误拒答 {s.get('错误拒答数', 0)}（计入）")
        for m in METRIC_NAMES:
            print(f"  {m}: {s.get(m)}（有效 {s.get(f'{m}_有效题数')}，null {s.get(f'{m}_null题数')}）")
        for _, label in _EFFICIENCY_KEYS:
            print(f"  {label}: {s.get(label)}")
    if "comparison" in report:
        comp_q = report["comparison"]["质量指标（diff=优化后-优化前，正值更优）"]
        print("\n===== 质量对比（diff=optimized-baseline，附配对差值95%CI）=====")
        for m, d in comp_q.items():
            ci = d.get("差值95%CI")
            extra = (f" | 配对均值={d.get('配对差值均值')} CI={ci} 显著={d.get('统计显著(CI不含0)')}"
                     if ci else "")
            print(f"{m}: baseline={d['baseline']} optimized={d['optimized']} diff={d['diff']}{extra}")
        comp = report["comparison"]["效率指标"]
        print("\n===== 效率对比（变化率为正 = 优化后更省/更快）=====")
        for label, d in comp.items():
            print(f"{label}: baseline={d['baseline']} optimized={d['optimized']} 变化率={d['变化率_正值更省']}")


if __name__ == "__main__":
    main()
