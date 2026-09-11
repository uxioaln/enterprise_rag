# 集成测试：GET /history/{session_id} 接口
import pytest


class TestHistoryAPI:

    def test_history_not_found(self, client):
        """查询不存在的 session_id 应返回 404"""
        response = client.get("/history/nonexistent-session")
        assert response.status_code == 404
        assert "未找到会话" in response.json()["detail"]

    def test_history_structure(self, client):
        """先发起一次问答，再查询历史，验证返回结构完整"""
        # 发起一次非流式问答（避免 SSE 时序问题）
        client.post(
            "/chat?stream=false",
            json={"session_id": "history-session", "question": "\"中芯国际\"的地位？"},
        )

        response = client.get("/history/history-session")
        assert response.status_code == 200
        body = response.json()

        assert body["session_id"] == "history-session"
        assert body["total"] == 1
        assert len(body["records"]) == 1

        record = body["records"][0]
        assert "question" in record
        assert "answer" in record
        assert "step_by_step_analysis" in record
        assert "reasoning_summary" in record
        assert "relevant_pages" in record
        assert "references" in record
        assert "elapsed_seconds" in record
        assert "created_at" in record

    def test_history_multiple_records(self, client):
        """同一会话多次问答后，历史应包含全部记录"""
        session_id = "multi-session"
        questions = [
            "\"中芯国际\"的营收情况如何？",
            "\"中芯国际\"的利润情况如何？",
            "\"中芯国际\"的产能情况如何？",
        ]
        for q in questions:
            client.post(
                "/chat?stream=false",
                json={"session_id": session_id, "question": q},
            )

        response = client.get(f"/history/{session_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 3
        record_questions = [r["question"] for r in body["records"]]
        assert questions[0] in record_questions
        assert questions[1] in record_questions
        assert questions[2] in record_questions
