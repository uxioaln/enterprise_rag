# 全局测试 fixtures：隔离的临时数据目录、Mock Pipeline、FastAPI TestClient
import csv
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router
from app.services import PipelineService, split_answer_chunks
from app.storage import InMemorySessionStorage


# ---------- 固定模拟数据 ----------

MOCK_ANSWER = {
    "step_by_step_analysis": "第一步：检索相关研报段落。第二步：提取中芯国际产能数据。第三步：综合分析得出结论。",
    "reasoning_summary": "中芯国际是晶圆代工龙头企业，产能利用率持续提升。",
    "relevant_pages": [
        {"file_name": "测试报告.pdf", "page": 3},
        {"file_name": "测试报告.pdf", "page": 5},
    ],
    "final_answer": "中芯国际是全球领先的集成电路晶圆代工企业。其产能利用率在2024年显著提升。国产化趋势不改，长期看好。",
    "references": [
        {"pdf_file_name": "测试报告.pdf", "page_index": 3},
        {"pdf_file_name": "测试报告.pdf", "page_index": 5},
    ],
}


# ---------- tmp_data_dir：隔离的临时数据目录 ----------

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """创建一个隔离的临时 data/stock_data 目录，含 pdf_reports 子目录和初始 subset.csv"""
    data_root = tmp_path / "stock_data"
    data_root.mkdir(parents=True)

    # pdf_reports 目录
    (data_root / "pdf_reports").mkdir()

    # databases 目录
    (data_root / "databases").mkdir()
    (data_root / "databases" / "chunked_reports").mkdir(parents=True)
    (data_root / "databases" / "vector_dbs").mkdir(parents=True)

    # debug_data 目录
    (data_root / "debug_data").mkdir()
    (data_root / "debug_data" / "03_reports_markdown").mkdir(parents=True)
    (data_root / "debug_data" / "03_reports_content_list").mkdir(parents=True)

    # 初始 subset.csv（含一条预置记录，方便公司路由测试）
    subset_path = data_root / "subset.csv"
    with open(subset_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "company_name", "sha1"])
        writer.writerow(["测试报告.pdf", "中芯国际", "stock_test001"])

    return data_root


# ---------- mock_pipeline：模拟整个 AI 处理 Pipeline ----------

@pytest.fixture
def mock_pipeline(tmp_data_dir: Path) -> MagicMock:
    """用 MagicMock 模拟 Pipeline，paths 指向临时目录，方法返回预设数据"""
    pipeline = MagicMock()

    # 配置 paths 属性，指向隔离的临时目录
    pipeline.paths = MagicMock()
    pipeline.paths.root_path = tmp_data_dir
    pipeline.paths.subset_path = tmp_data_dir / "subset.csv"
    pipeline.paths.pdf_reports_dir = tmp_data_dir / "pdf_reports"
    pipeline.paths.databases_path = tmp_data_dir / "databases"
    pipeline.paths.vector_db_dir = tmp_data_dir / "databases" / "vector_dbs"
    pipeline.paths.documents_dir = tmp_data_dir / "databases" / "chunked_reports"
    pipeline.paths.debug_data_path = tmp_data_dir / "debug_data"
    pipeline.paths.reports_markdown_path = tmp_data_dir / "debug_data" / "03_reports_markdown"
    pipeline.paths.reports_content_list_path = tmp_data_dir / "debug_data" / "03_reports_content_list"

    # answer_single_question 返回预设结构化答案
    pipeline.answer_single_question.return_value = dict(MOCK_ANSWER)

    # export_reports_to_markdown / chunk_reports 默认为 no-op（MagicMock 自动返回 MagicMock）
    # _vectorize_single_report 在 PipelineService 中，不经过 pipeline

    return pipeline


# ---------- mock_storage：干净的内存存储 ----------

@pytest.fixture
def mock_storage() -> InMemorySessionStorage:
    """返回一个干净的异步内存存储实例"""
    return InMemorySessionStorage()


# ---------- client：FastAPI TestClient，注入 mock_pipeline ----------

@pytest.fixture
def client(mock_pipeline: MagicMock, mock_storage: InMemorySessionStorage) -> TestClient:
    """创建 TestClient，绕过真实 lifespan，直接注入 mock_pipeline 和 mock_storage"""
    # 构建测试专用 app，不挂载 main.py 的 lifespan（避免初始化真实 Pipeline）
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.state.pipeline_service = PipelineService(mock_pipeline)
    test_app.state.storage = mock_storage

    with TestClient(test_app) as c:
        yield c


# ---------- Celery / AsyncOpenAI 等外部依赖的全局 Mock ----------

@pytest.fixture(autouse=True)
def mock_celery_task(monkeypatch):
    """自动 Mock 掉 Celery 任务提交（app.api.parse_and_index_pdf），避免测试连接真实 Redis broker。

    delay() 返回带固定 task_id 的 Mock 对象，供 /upload 的 202 响应断言使用；
    返回 mock 本身，供个别测试配置 delay 的异常行为（如提交失败场景）。
    """
    mock_task = MagicMock()
    mock_task.delay.return_value = MagicMock(id="test-task-id")
    monkeypatch.setattr("app.api.parse_and_index_pdf", mock_task)
    return mock_task


@pytest.fixture(autouse=True)
def mock_async_openai_stream(monkeypatch):
    """自动 Mock AsyncOpenAI 流式客户端，使 SSE delta 按句子切分原样下发。

    PipelineService.stream_final_answer 通过 AsyncOpenAI.chat.completions.create(stream=True)
    拿到流迭代器并直接访问 chunk.choices[0].delta.content。此处用 fake 客户端模拟该流：
    把请求 messages 中最后一条 user 内容（即 final_answer）按 split_answer_chunks 切块，
    逐 chunk 作为 delta.content 返回，保证 delta 拼接后等于 final_answer。
    """

    class _FakeDelta:
        def __init__(self, content):
            self.content = content

    class _FakeChoice:
        def __init__(self, content):
            self.delta = _FakeDelta(content)

    class _FakeChunk:
        def __init__(self, content):
            self.choices = [_FakeChoice(content)]

    class _FakeStream:
        """模拟 openai.AsyncStream：异步迭代产出 delta chunk，支持 async with"""
        def __init__(self, user_content):
            chunks = split_answer_chunks(user_content)
            # 切分为空（如纯空内容）时整体兜底为原始内容，保证至少能产出一次
            self._chunks = chunks if chunks else [user_content]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._chunks:
                raise StopAsyncIteration
            return _FakeChunk(self._chunks.pop(0))

        async def __aenter__(self):
            # async with stream: 进入时返回自身，与真实 AsyncStream 行为一致
            return self

        async def __aexit__(self, exc_type, exc, tb):
            # 退出时关闭流，确保资源清理逻辑被覆盖
            await self.close()
            return False

        async def close(self):
            pass

    class _FakeCompletions:
        async def create(self, *, model=None, messages=None, stream=False, **kwargs):
            # 取最后一条 user 消息内容作为待流式输出的原文（即 final_answer）
            user_content = ""
            for m in (messages or []):
                if m.get("role") == "user":
                    user_content = m.get("content", "")
            return _FakeStream(user_content)

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = _FakeChat()

    def _fake_get_async_client(self):
        return _FakeAsyncOpenAI()

    # 替换 PipelineService 的懒加载客户端构造方法，避免真实网络调用
    monkeypatch.setattr(PipelineService, "_get_async_client", _fake_get_async_client)


@pytest.fixture
def make_mock_pdf_bytes():
    """返回一个生成最小 PDF 字节的工厂函数（仅含 PDF 头尾，用于上传测试）"""
    def _make(size_bytes: int = 256) -> bytes:
        # PDF 最小结构：header + 少量 body + EOF
        header = b"%PDF-1.4\n"
        body = b"0 obj\n<< /Type /Catalog >>\nendobj\n"
        eof = b"%%EOF\n"
        padding = b"\x00" * max(0, size_bytes - len(header) - len(body) - len(eof))
        return header + body + padding + eof

    return _make
