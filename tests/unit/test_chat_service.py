# 单元测试：process_chat 流式问答逻辑与答案切分
import asyncio
import json
from unittest.mock import MagicMock

import pytest

from app.services import PipelineService, format_sse, process_chat, split_answer_chunks
from app.storage import InMemorySessionStorage


class TestSplitAnswerChunks:
    """测试答案文本切分逻辑"""

    def test_split_by_chinese_sentence(self):
        """按中文句号切分"""
        text = "第一句。第二句！第三句？"
        chunks = split_answer_chunks(text)
        assert len(chunks) == 3
        assert "第一句" in chunks[0]
        assert "第二句" in chunks[1]
        assert "第三句" in chunks[2]

    def test_split_empty_text(self):
        """空文本返回空列表"""
        assert split_answer_chunks("") == []

    def test_long_sentence_hard_split(self):
        """超长句应按 max_chars 硬切"""
        text = "a" * 200 + "。"
        chunks = split_answer_chunks(text, max_chars=60)
        assert len(chunks) > 1
        assert all(len(c) <= 60 for c in chunks)

    def test_no_terminator(self):
        """无结束符的文本应整体作为一个块"""
        chunks = split_answer_chunks("无结束符的文本")
        assert len(chunks) == 1
        assert chunks[0] == "无结束符的文本"


class TestFormatSSE:
    """测试 SSE 消息格式化"""

    def test_format_sse_structure(self):
        """SSE 消息应包含 event 行、data 行和空行结尾"""
        msg = format_sse("delta", {"content": "测试"})
        assert msg.startswith("event: delta\n")
        assert "data: " in msg
        assert msg.endswith("\n\n")

    def test_format_sse_json_chinese(self):
        """SSE data 中的中文不应被转义"""
        msg = format_sse("delta", {"content": "中文测试"})
        assert "中文测试" in msg
        assert "\\u" not in msg


class TestProcessChat:
    """测试 process_chat 异步生成器的 SSE 事件流"""

    @pytest.fixture
    def service(self, mock_pipeline):
        return PipelineService(mock_pipeline)

    @pytest.fixture
    def storage(self):
        return InMemorySessionStorage()

    @pytest.mark.asyncio
    async def test_event_sequence_start_reasoning_delta_done(self, service, storage, mock_pipeline):
        """SSE 事件流顺序应为 start -> reasoning -> delta(多次) -> done"""
        events = []
        async for msg in process_chat(service, storage, "s1", "测试问题"):
            events.append(parse_single_sse(msg))

        event_names = [e["event"] for e in events]
        # 首个事件为 start
        assert event_names[0] == "start"
        # reasoning 在 delta 之前
        reasoning_idx = event_names.index("reasoning")
        first_delta_idx = event_names.index("delta")
        assert reasoning_idx < first_delta_idx
        # 最后一个事件为 done
        assert event_names[-1] == "done"
        # delta 出现至少一次
        assert event_names.count("delta") >= 1

    @pytest.mark.asyncio
    async def test_start_event_contains_session_and_question(self, service, storage):
        """start 事件应包含 session_id 和 question"""
        events = []
        async for msg in process_chat(service, storage, "my-session", "什么是中芯国际？"):
            events.append(parse_single_sse(msg))

        start_data = events[0]["data"]
        assert start_data["session_id"] == "my-session"
        assert start_data["question"] == "什么是中芯国际？"

    @pytest.mark.asyncio
    async def test_done_event_contains_full_answer(self, service, storage, mock_pipeline):
        """done 事件应包含完整答案、相关页面和引用"""
        events = []
        async for msg in process_chat(service, storage, "s1", "问题"):
            events.append(parse_single_sse(msg))

        done_data = events[-1]["data"]
        assert events[-1]["event"] == "done"
        assert "answer" in done_data
        assert "relevant_pages" in done_data
        assert "references" in done_data
        assert done_data["answer"] == "中芯国际是全球领先的集成电路晶圆代工企业。其产能利用率在2024年显著提升。国产化趋势不改，长期看好。"

    @pytest.mark.asyncio
    async def test_history_persisted_after_chat(self, service, storage):
        """问答完成后应将会话记录写入存储"""
        async for _ in process_chat(service, storage, "persist-session", "问题"):
            pass

        history = await storage.get_history("persist-session")
        assert len(history) == 1
        assert history[0]["question"] == "问题"

    @pytest.mark.asyncio
    async def test_error_event_on_pipeline_failure(self, service, storage, mock_pipeline):
        """Pipeline 推理失败时应推送 error 事件而非崩溃"""
        mock_pipeline.answer_single_question.side_effect = RuntimeError("模型超时")
        events = []
        async for msg in process_chat(service, storage, "s1", "问题"):
            events.append(parse_single_sse(msg))

        event_names = [e["event"] for e in events]
        assert "error" in event_names
        error_data = [e for e in events if e["event"] == "error"][0]["data"]
        # 业务异常 type 应为 business，且 message 含可读的异常信息
        assert error_data["type"] == "business"
        assert "模型超时" in error_data["message"]


# ---------- 辅助函数 ----------

def parse_single_sse(raw: str) -> dict:
    """解析单条 SSE 消息，返回 {event, data}"""
    event = None
    data_str = None
    for line in raw.strip().split("\n"):
        if line.startswith("event: "):
            event = line[len("event: "):]
        elif line.startswith("data: "):
            data_str = line[len("data: "):]
    return {"event": event, "data": json.loads(data_str) if data_str else {}}
