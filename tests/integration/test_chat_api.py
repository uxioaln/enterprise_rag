# 集成测试：POST /chat 接口（SSE 流式 + 非流式）
import json

import pytest

from tests.conftest import MOCK_ANSWER


def parse_sse(raw_text: str):
    """解析完整的 SSE 响应文本，返回事件列表 [{event, data}, ...]"""
    events = []
    for block in raw_text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = None
        data_obj = {}
        for line in block.strip().split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: "):
                data_obj = json.loads(line[len("data: "):])
        events.append({"event": event_name, "data": data_obj})
    return events


class TestChatAPI:

    def test_chat_stream_success(self, client, mock_pipeline):
        """流式问答：返回 200，SSE 事件顺序 start -> reasoning -> delta -> done"""
        response = client.post(
            "/chat",
            json={"session_id": "stream-session", "question": "\"中芯国际\"的地位如何？"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        events = parse_sse(response.text)
        event_names = [e["event"] for e in events]

        # 验证事件顺序
        assert event_names[0] == "start"
        assert "reasoning" in event_names
        assert "delta" in event_names
        assert event_names[-1] == "done"

        # reasoning 在 delta 之前
        assert event_names.index("reasoning") < event_names.index("delta")

        # done 事件包含完整答案与引用
        done_data = events[-1]["data"]
        assert "answer" in done_data
        assert "relevant_pages" in done_data
        assert "references" in done_data
        assert len(done_data["relevant_pages"]) > 0

    def test_chat_non_stream(self, client, mock_pipeline):
        """非流式问答：stream=false 时返回标准 JSON"""
        response = client.post(
            "/chat?stream=false",
            json={"session_id": "json-session", "question": "\"中芯国际\"的产能如何？"},
        )
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

        body = response.json()
        assert body["session_id"] == "json-session"
        assert "answer" in body
        assert "step_by_step_analysis" in body
        assert "reasoning_summary" in body
        assert body["answer"] == MOCK_ANSWER["final_answer"]

    def test_chat_history_persisted_after_stream(self, client):
        """流式问答后，会话历史应被正确保存"""
        client.post(
            "/chat",
            json={"session_id": "persist-session", "question": "\"中芯国际\"的营收？"},
        )
        # 通过 /history 接口验证历史已正确持久化（异步存储统一经接口读取）
        resp = client.get("/history/persist-session")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert "营收" in body["records"][0]["question"]
        assert body["records"][0]["answer"] == MOCK_ANSWER["final_answer"]

    def test_chat_history_persisted_after_non_stream(self, client):
        """非流式问答后，会话历史也应被正确保存"""
        client.post(
            "/chat?stream=false",
            json={"session_id": "persist-json-session", "question": "\"中芯国际\"的利润？"},
        )
        resp = client.get("/history/persist-json-session")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["records"][0]["answer"] == MOCK_ANSWER["final_answer"]

    def test_chat_error_on_pipeline_failure(self, client, mock_pipeline):
        """Pipeline 推理失败时 SSE 应推送 error 事件"""
        mock_pipeline.answer_single_question.side_effect = RuntimeError("模型不可用")
        response = client.post(
            "/chat",
            json={"session_id": "err-session", "question": "\"中芯国际\"的整体情况如何？"},
        )
        assert response.status_code == 200  # SSE 本身 200，错误在事件内
        events = parse_sse(response.text)
        event_names = [e["event"] for e in events]
        assert "error" in event_names
        assert event_names[0] == "start"  # start 仍先发送

    def test_chat_delta_events_contain_answer_chunks(self, client):
        """delta 事件拼接后应等于完整答案"""
        response = client.post(
            "/chat",
            json={"session_id": "delta-session", "question": "\"中芯国际\"的整体情况如何？"},
        )
        events = parse_sse(response.text)
        delta_chunks = [e["data"]["content"] for e in events if e["event"] == "delta"]
        combined = "".join(delta_chunks)
        assert combined == MOCK_ANSWER["final_answer"]
