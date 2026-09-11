# 业务逻辑适配层：连接 FastAPI 路由与 src.pipeline 中的 RAG 流水线
import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from fastapi import Request
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError

from src.ingestion import VectorDBIngestor
from src.pipeline import Pipeline

logger = logging.getLogger(__name__)

# SSE 单条消息格式：event + data（JSON），以空行结尾
_SSE_TEMPLATE = "event: {event}\ndata: {data}\n\n"

# 流式输出最终答案时使用的系统提示：要求模型逐字原样输出最终答案，
# 不添加任何前缀/后缀/解释，保证 SSE delta 拼接后等于 Pipeline 产出的 final_answer
_STREAM_ANSWER_SYSTEM_PROMPT = (
    "你是一个文本输出助手。请将用户提供的文本逐字、完整地原样输出，"
    "不得添加任何前缀、后缀、解释、问候、总结或格式化标记，也不得修改或增删任何标点，仅输出原文。"
)

# SSE keep-alive 注释帧：长时间无数据时下发，维持连接不断开
_KEEPALIVE_FRAME = ": keep-alive\n\n"
# 无数据时下发 keep-alive 的间隔（秒）
_KEEP_ALIVE_INTERVAL = 30
# 单次 SSE 响应总超时（秒），超时后主动关闭连接
_TOTAL_TIMEOUT = 300
# 每推送 N 个 delta chunk 检测一次客户端是否已断开
_DISCONNECT_CHECK_EVERY = 4

# 按中文句子结束符切分（保留分隔符本身）
_SENTENCE_RE = re.compile(r"[^。！？；!?\n]+[。！？；!?\n]?")


def format_sse(event: str, data: dict) -> str:
    """将事件名与数据格式化为一条 SSE 消息"""
    return _SSE_TEMPLATE.format(event=event, data=json.dumps(data, ensure_ascii=False))


def split_answer_chunks(text: str, max_chars: int = 60) -> List[str]:
    """把完整答案文本切分为逐句的小块，用于 SSE 流式输出。

    优先按句子切分；超过 max_chars 的长句再做硬切，避免单块过大。
    """
    if not text:
        return []
    chunks: List[str] = []
    for sentence in _SENTENCE_RE.findall(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        # 超长句按 max_chars 硬切
        while len(sentence) > max_chars:
            chunks.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if sentence:
            chunks.append(sentence)
    return chunks


class PipelineService:
    """封装 Pipeline 的同步调用，提供异步入口供路由层使用"""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        # 懒加载的 AsyncOpenAI 客户端，用于 SSE 流式输出最终答案
        self._async_client: AsyncOpenAI | None = None

    @property
    def pdf_reports_dir(self) -> Path:
        """PDF 报告保存目录（data/stock_data/pdf_reports）"""
        return self.pipeline.paths.pdf_reports_dir

    # ---------- 问答 ----------

    def answer_question_sync(self, question: str, kind: str = "string") -> dict:
        """同步调用 Pipeline 单问推理，返回结构化答案（在线程池中执行）"""
        return self.pipeline.answer_single_question(question, kind=kind)

    async def answer_question(self, question: str, kind: str = "string") -> dict:
        """异步包装：把同步的 Pipeline 推理放到线程池，避免阻塞事件循环"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.answer_question_sync(question, kind))

    # ---------- 流式输出（基于 openai 原生异步流式迭代器） ----------

    def _get_async_client(self) -> AsyncOpenAI:
        """懒构造 AsyncOpenAI 客户端，复用 AGICTO 平台的 OpenAI 兼容接口"""
        if self._async_client is None:
            load_dotenv()
            self._async_client = AsyncOpenAI(
                api_key=os.getenv("AGICTO_API_KEY"),
                base_url="https://api.agicto.cn/v1",
            )
        return self._async_client

    async def stream_final_answer(self, final_answer: str) -> AsyncGenerator[str, None]:
        """使用 openai 库原生异步流式迭代器逐 chunk 输出最终答案。

        直接调用 AsyncOpenAI.chat.completions.create(stream=True) 拿到流迭代器，
        遍历时直接读取 chunk.choices[0].delta.content 作为增量文本 yield 出去。
        严禁手动解析 "data: {...}" SSE 字符串、严禁 str(chunk).split / json.loads 处理原始 SSE。
        使用 ``async with stream:`` 包裹流迭代器，确保底层 HTTP 连接被归还到连接池，
        即使发生异常或外部取消（CancelledError）也能正确关闭流。
        """
        if not final_answer:
            return
        client = self._get_async_client()
        # 注意：stream=True 返回的是 AsyncStream，await 后得到流迭代器
        stream = await client.chat.completions.create(
            model=self.pipeline.run_config.answering_model,
            messages=[
                {"role": "system", "content": _STREAM_ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": final_answer},
            ],
            stream=True,
        )
        # 使用 async with 包裹：无论正常退出、异常或取消，都会调用 stream.close()
        # 归还底层 HTTP 连接到连接池，避免连接泄漏
        async with stream:
            # 直接访问 delta.content，不解析原始 SSE 行
            async for chunk in stream:
                # 部分兼容服务在末尾会下发空 choices 的 usage chunk，需跳过
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    async def aclose(self) -> None:
        """释放异步客户端等资源，供应用关闭（lifespan shutdown）时调用。

        关闭懒加载的 AsyncOpenAI 客户端，归还其底层 HTTP 连接池；多次调用安全。
        """
        if self._async_client is not None:
            with suppress(Exception):
                await self._async_client.close()
            self._async_client = None
            logger.info("AsyncOpenAI 客户端已关闭，HTTP 连接池已释放")

    # ---------- 文档上传入库 ----------

    async def ingest_pdf(self, file_name: str, company_name: str) -> dict:
        """异步包装：对新上传的 PDF 执行解析、分块、向量化入库"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.ingest_pdf_sync(file_name, company_name))

    def register_pdf(self, file_name: str, company_name: str) -> dict:
        """登记 subset 条目（POST /upload 异步化的前置步骤，供 API 层调用）。

        解析、分块、向量化等耗时步骤已交由 Celery 任务 parse_and_index_pdf 执行，
        API 层仅完成 subset.csv 登记（取得 sha1）。MinerU 解析改为签名 URL 直传本地文件，
        无需再上传 OSS。
        """
        sha1 = self._ensure_subset_entry(file_name, company_name)
        return {"file_name": file_name, "company_name": company_name, "sha1": sha1}

    def ingest_pdf_sync(self, file_name: str, company_name: str) -> dict:
        """对新上传的 PDF 执行完整入库流程（同步，在线程池中执行）。

        流程：
        1. 在 subset.csv 中登记文件条目（file_name / company_name / sha1）
        2. 调用 Pipeline.export_reports_to_markdown 做 MinerU 云端解析（签名 URL 直传本地文件）
        3. 调用 Pipeline.chunk_reports 基于真实页码分块
        4. 增量向量化：仅对新文件的分块生成 FAISS 索引（复用 VectorDBIngestor），
           避免每次上传都全量重建所有索引、重复消耗嵌入调用
        """
        t0 = time.time()
        sha1 = self._ensure_subset_entry(file_name, company_name)

        print(f"[上传] 开始 MinerU 云端解析: {file_name}")
        self.pipeline.export_reports_to_markdown(file_name)

        print(f"[上传] 开始报告分块: {file_name}")
        self.pipeline.chunk_reports()

        print(f"[上传] 开始增量向量化入库: {file_name} (sha1={sha1})")
        self._vectorize_single_report(file_name)

        elapsed = time.time() - t0
        print(f"[上传] 入库完成: {file_name}，总耗时 {elapsed:.2f} 秒")
        return {"file_name": file_name, "company_name": company_name, "sha1": sha1}

    def _ensure_subset_entry(self, file_name: str, company_name: str) -> str:
        """确保 subset.csv 中存在该文件的条目，返回其 sha1。

        已存在则复用原 sha1；不存在则追加一行，sha1 用 stock_ + uuid 生成。
        """
        subset_path: Path = self.pipeline.paths.subset_path
        if subset_path.exists():
            try:
                df = pd.read_csv(subset_path, encoding="utf-8")
            except UnicodeDecodeError:
                # 与 src 内部保持一致的编码兼容策略
                df = pd.read_csv(subset_path, encoding="gbk")
        else:
            df = pd.DataFrame(columns=["file_name", "company_name", "sha1"])

        # 已登记的文件直接复用 sha1
        existing = df[df["file_name"] == file_name]
        if not existing.empty:
            return str(existing.iloc[0]["sha1"])

        sha1 = f"stock_{uuid.uuid4().hex[:12]}"
        new_row = pd.DataFrame([{"file_name": file_name, "company_name": company_name, "sha1": sha1}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(subset_path, index=False, encoding="utf-8")
        print(f"[上传] 已在 subset.csv 登记新文件: {file_name} -> {sha1}")
        return sha1

    def _vectorize_single_report(self, file_name: str) -> None:
        """只对指定文件新产出的分块 JSON 生成 FAISS 索引。

        复用 VectorDBIngestor.process_reports 的公开逻辑：把新文件的分块
        放入临时目录，索引直接写入既有 vector_db_dir。
        """
        stem = Path(file_name).stem
        chunk_file: Path = self.pipeline.paths.documents_dir / f"{stem}.json"
        if not chunk_file.exists():
            raise FileNotFoundError(f"未找到分块结果: {chunk_file}，请检查解析与分块步骤是否成功")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_chunk = Path(tmp_dir) / chunk_file.name
            shutil.copy(chunk_file, tmp_chunk)
            ingestor = VectorDBIngestor()
            ingestor.process_reports(Path(tmp_dir), self.pipeline.paths.vector_db_dir)

    def vectorize_single_report(self, file_name: str) -> None:
        """公开包装：供 Celery 任务（app.tasks.parse_and_index_pdf）复用单文件增量向量化逻辑"""
        return self._vectorize_single_report(file_name)


async def process_chat(
    service: PipelineService,
    storage,
    session_id: str,
    question: str,
    request: Request | None = None,
) -> AsyncGenerator[str, None]:
    """核心流式问答：异步生成器，逐块 yield SSE 消息。

    推理阶段调用 Pipeline.answer_single_question（同步阻塞，放入线程池执行）拿到
    结构化答案；随后将结构化推理结果分阶段映射到 SSE 事件，最终答案则直接基于
    openai 库原生异步流式迭代器（AsyncOpenAI + stream=True）逐 chunk 推送，
    不手动拼接或解析 SSE 字符串。

    连接生命周期管理：
      - 客户端断开检测：注入 Request 对象，每推送 _DISCONNECT_CHECK_EVERY 个 delta
        以及 keep-alive 超时时检测 request.is_disconnected()，断开后立即取消模型流，
        避免继续消耗 token。
      - keep-alive：若 _KEEP_ALIVE_INTERVAL 秒内无数据，下发 SSE 注释帧维持连接。
      - 总超时：超过 _TOTAL_TIMEOUT 秒主动关闭连接并推送 timeout error。
      - 分层异常处理：rate_limit / timeout / business / internal 四类 error 事件。

    事件流：
      start     -> 会话与问题信息
      reasoning -> 分步推理与推理摘要
      delta     -> 最终答案的逐 chunk 内容（多次，来自 openai 原生流 delta.content）
      done      -> 完整结果（含相关页面 / 引用），并写入会话历史
      error     -> 推理或流式调用异常信息（含 type 字段区分类型）
    """
    yield format_sse("start", {"session_id": session_id, "question": question})

    t0 = time.time()
    deadline = t0 + _TOTAL_TIMEOUT

    # ---- 阶段一：推理（受总超时约束） ----
    try:
        remaining = max(1.0, deadline - time.time())
        answer_dict = await asyncio.wait_for(
            service.answer_question(question), timeout=remaining
        )
    except asyncio.TimeoutError:
        yield format_sse("error", {"type": "timeout", "message": "模型服务响应超时，请重试"})
        return
    except RateLimitError:
        yield format_sse(
            "error", {"type": "rate_limit", "message": "当前请求过于频繁，请稍后再试"}
        )
        return
    except (APITimeoutError, APIConnectionError):
        yield format_sse("error", {"type": "timeout", "message": "模型服务响应超时，请重试"})
        return
    except Exception as e:
        # 业务异常：检索不到文档、FAISS 索引未找到等，向客户端暴露可读信息
        logger.exception("answer_question failed")
        yield format_sse("error", {"type": "business", "message": f"推理失败: {e}"})
        return

    elapsed = time.time() - t0

    # ---- 重试循环元数据推送（仅在启用重试循环且发生多轮时推送）----
    retry_meta = answer_dict.get("retry_metadata")
    if retry_meta and isinstance(retry_meta, dict) and retry_meta.get("total_attempts", 1) > 1:
        rounds = retry_meta.get("rounds", [])
        # retry_start：重试循环概览
        yield format_sse("retry_start", {
            "total_attempts": retry_meta.get("total_attempts", len(rounds)),
            "forced_exit": retry_meta.get("forced_exit", False),
        })
        # retry_reasoning：每轮的查询、置信度与反思
        for r in rounds:
            yield format_sse("retry_reasoning", {
                "attempt": r.get("attempt", 0),
                "query": r.get("query", ""),
                "confidence": r.get("confidence", {}),
                "critique": r.get("critique", ""),
                "strategy": r.get("strategy", ""),
                "rationale": r.get("rationale", ""),
            })
        # retry_success：最终采纳的置信度
        final_conf = retry_meta.get("final_confidence", {})
        yield format_sse("retry_success", {
            "final_confidence": final_conf,
            "forced_exit": retry_meta.get("forced_exit", False),
        })

    # 结构化推理结果整体下发（推理已就绪即推送，不等最终答案生成完再一次性推）
    yield format_sse(
        "reasoning",
        {
            "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
            "reasoning_summary": answer_dict.get("reasoning_summary", ""),
        },
    )

    # ---- 阶段二：流式输出最终答案（生产者/消费者 + keep-alive + 断开检测 + 总超时） ----
    final_answer = str(answer_dict.get("final_answer", ""))

    # 生产者：迭代 stream_final_answer，把 delta 放入队列；异常转化为 error 项
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    async def _producer() -> None:
        """消费 openai 原生流，把 delta 或 error 事件投递到队列"""
        try:
            async for delta in service.stream_final_answer(final_answer):
                await queue.put(("delta", delta))
        except RateLimitError:
            await queue.put(
                ("error", {"type": "rate_limit", "message": "当前请求过于频繁，请稍后再试"})
            )
        except (APITimeoutError, APIConnectionError):
            await queue.put(("error", {"type": "timeout", "message": "模型服务响应超时，请重试"}))
        except Exception:
            # 未预料异常：记录堆栈，不向客户端暴露原始异常信息
            logger.exception("stream_final_answer failed")
            await queue.put(("error", {"type": "internal", "message": "系统内部错误"}))
        else:
            # 正常结束：投递哨兵通知消费者退出
            await queue.put(_SENTINEL)

    producer_task = asyncio.create_task(_producer())
    stream_completed = False
    delta_count = 0

    try:
        while True:
            # 检查总超时
            remaining = deadline - time.time()
            if remaining <= 0:
                yield format_sse("error", {"type": "timeout", "message": "模型服务响应超时，请重试"})
                break

            # 等待下一条数据，最长等待 min(keep-alive 间隔, 剩余时间)
            wait_time = min(_KEEP_ALIVE_INTERVAL, remaining)
            try:
                item = await asyncio.wait_for(queue.get(), timeout=wait_time)
            except asyncio.TimeoutError:
                # 超时内无数据：检测客户端是否已断开，否则下发 keep-alive 注释帧
                if request is not None and await request.is_disconnected():
                    break
                yield _KEEPALIVE_FRAME
                continue

            if item is _SENTINEL:
                stream_completed = True
                break

            event_type, payload = item
            if event_type == "delta":
                yield format_sse("delta", {"content": payload})
                delta_count += 1
                # 每推送 N 个 chunk 检测一次客户端是否已断开
                if delta_count % _DISCONNECT_CHECK_EVERY == 0 and request is not None:
                    if await request.is_disconnected():
                        break
            elif event_type == "error":
                yield format_sse("error", payload)
                break
    finally:
        # 消费者退出时取消生产者任务，触发 stream_final_answer 的 async with 清理，
        # 归还 openai 底层 HTTP 连接到连接池
        if not producer_task.done():
            producer_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await producer_task

    # 仅在流式正常完成时写入历史并推送 done；断开/超时/异常不推送 done
    if not stream_completed:
        return

    # 写入会话历史（异步存储：完整保留推理与引用字段 + 重试元数据）
    await storage.append_record(
        session_id,
        question,
        answer_dict.get("final_answer", ""),
        step_by_step_analysis=answer_dict.get("step_by_step_analysis", ""),
        reasoning_summary=answer_dict.get("reasoning_summary", ""),
        relevant_pages=answer_dict.get("relevant_pages", []),
        references=answer_dict.get("references", []),
        elapsed_seconds=elapsed,
        confidence=answer_dict.get("confidence"),
        retry_metadata=answer_dict.get("retry_metadata"),
        forced_exit=bool(answer_dict.get("retry_metadata", {}).get("forced_exit", False)) if isinstance(answer_dict.get("retry_metadata"), dict) else False,
    )

    yield format_sse(
        "done",
        {
            "session_id": session_id,
            "question": question,
            "answer": final_answer,
            "relevant_pages": answer_dict.get("relevant_pages", []),
            "references": answer_dict.get("references", []),
            "elapsed_seconds": round(elapsed, 2),
            "confidence": answer_dict.get("confidence"),
            "retry_metadata": answer_dict.get("retry_metadata"),
            "forced_exit": bool(answer_dict.get("retry_metadata", {}).get("forced_exit", False)) if isinstance(answer_dict.get("retry_metadata"), dict) else False,
        },
    )
