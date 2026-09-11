# app/tasks.py
"""Celery 异步任务：PDF 解析（MinerU）-> 分块 -> 向量化入库。

任务 parse_and_index_pdf 由 POST /upload 提交（文件保存与 subset 登记在 API 层完成），
内部完全复用 src/pipeline.py、src/text_splitter.py、src/ingestion.py 的既有逻辑：
  1. pipeline.export_reports_to_markdown：MinerU 云端解析（签名 URL 直传本地文件）并抽取 content_list/images
  2. pipeline.chunk_reports：基于真实页码分块
  3. PipelineService.vectorize_single_report：增量向量化（FAISS 索引加锁 + 原子落盘）

异常处理：任一步骤失败时 self.retry(exc=exc, countdown=10)，最多重试 3 次，
最终失败后任务状态为 FAILURE，可经 GET /tasks/{task_id} 查询。
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from pyprojroot import here

from app.celery_app import celery_app

logger = logging.getLogger(__name__)


def _parse_chunk_index(file_name: str) -> None:
    """执行解析、分块、向量化入库的完整流程（复用现有 Pipeline 逻辑）。"""
    # 延迟导入：避免 celery_app 模块加载时引入 FAISS / openai 等重依赖
    from app.services import PipelineService
    from src.pipeline import Pipeline, max_config

    # 数据根目录与 app.main 保持一致：data/stock_data
    root_path = here() / "data" / "stock_data"
    pipeline = Pipeline(root_path, run_config=max_config)
    service = PipelineService(pipeline)

    # MinerU 同步 HTTP 调用（轮询等待）与分块为阻塞操作：用线程池包裹执行，
    # 避免长时间占用 worker 主线程；分块依赖解析产物，故顺序提交
    with ThreadPoolExecutor(max_workers=1) as pool:
        parse_future = pool.submit(pipeline.export_reports_to_markdown, file_name)
        parse_future.result()
        chunk_future = pool.submit(pipeline.chunk_reports)
        chunk_future.result()

    # 增量向量化：内部经 VectorDBIngestor.add_chunks_to_index_atomic 加锁 + 原子落盘
    service.vectorize_single_report(file_name)


@celery_app.task(bind=True, name="app.tasks.parse_and_index_pdf", max_retries=3)
def parse_and_index_pdf(self, company_name: str, file_name: str, sha1: str) -> dict:
    """异步解析并索引新上传的 PDF 研报。

    参数（由 POST /upload 提交）：
        company_name: 文件所属公司名
        file_name: PDF 文件名（已保存在 data/stock_data/pdf_reports/ 下，
                   MinerU 解析通过签名 URL 直传该本地文件）
        sha1: subset.csv 中登记的文件标识（stock_ 前缀）

    返回：结果字典（存入 Redis result backend，供 GET /tasks/{task_id} 查询）。
    失败时以 10 秒间隔重试，最多 3 次，最终失败置为 FAILURE。
    """
    logger.info(
        "[parse_and_index_pdf] 开始处理: file=%s, company=%s, sha1=%s (attempt %s)",
        file_name, company_name, sha1, self.request.retries + 1,
    )
    try:
        _parse_chunk_index(file_name)
    except Exception as exc:
        logger.error(
            "[parse_and_index_pdf] 处理失败（第 %s 次）: file=%s, error=%s",
            self.request.retries + 1, file_name, exc,
        )
        # 失败重试：10 秒后再次执行，最多 3 次；超出后任务置为 FAILURE
        raise self.retry(exc=exc, countdown=10)

    result = {
        "file_name": file_name,
        "company_name": company_name,
        "sha1": sha1,
        "message": "MinerU 解析、分块与向量化入库完成",
    }
    logger.info("[parse_and_index_pdf] 处理成功: file=%s, sha1=%s", file_name, sha1)
    return result
