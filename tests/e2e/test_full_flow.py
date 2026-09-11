# E2E 测试：完整用户流程 上传 -> 问答 -> 查询历史
import json

import pandas as pd


class TestFullFlowE2E:
    """端到端用户故事：上传 PDF -> 发起问答 -> 查询历史"""

    def test_full_flow(self, client, mock_pipeline, tmp_data_dir, make_mock_pdf_bytes):
        """验证从上传到问答再到历史的完整流程连贯性"""
        # ---- 第一步：上传 PDF（异步化：仅受理，返回 202 + task_id）----
        pdf_bytes = make_mock_pdf_bytes(512)
        file_name = "中芯国际深度报告.pdf"
        upload_resp = client.post(
            "/upload",
            files={"file": (file_name, pdf_bytes, "application/pdf")},
            data={"company_name": "中芯国际"},
        )
        assert upload_resp.status_code == 202
        upload_body = upload_resp.json()
        assert upload_body["status"] == "pending"
        assert upload_body["task_id"] == "test-task-id"

        # 受理前置步骤同步完成：文件已落盘
        assert (tmp_data_dir / "pdf_reports" / file_name).exists()

        # 验证 subset.csv 已追加
        df = pd.read_csv(tmp_data_dir / "subset.csv", encoding="utf-8")
        assert file_name in df["file_name"].values

        # ---- 第二步：发起流式问答 ----
        session_id = "e2e-flow-session"
        chat_resp = client.post(
            "/chat",
            json={"session_id": session_id, "question": "\"中芯国际\"在晶圆制造行业中的地位如何？"},
        )
        assert chat_resp.status_code == 200
        assert "text/event-stream" in chat_resp.headers.get("content-type", "")

        # 解析 SSE 事件
        events = []
        for block in chat_resp.text.strip().split("\n\n"):
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

        event_names = [e["event"] for e in events]
        assert event_names[0] == "start"
        assert "reasoning" in event_names
        assert "delta" in event_names
        assert event_names[-1] == "done"

        # done 事件中的答案不为空
        done_data = events[-1]["data"]
        assert len(done_data["answer"]) > 0
        assert len(done_data["relevant_pages"]) > 0

        # ---- 第三步：查询历史记录 ----
        history_resp = client.get(f"/history/{session_id}")
        assert history_resp.status_code == 200
        history_body = history_resp.json()

        assert history_body["session_id"] == session_id
        assert history_body["total"] == 1
        record = history_body["records"][0]
        assert "中芯国际" in record["question"]
        assert record["answer"] == done_data["answer"]
        assert "created_at" in record

        # ---- 第四步：再次问答，验证历史累积 ----
        client.post(
            "/chat?stream=false",
            json={"session_id": session_id, "question": "\"中芯国际\"的产能利用率如何？"},
        )

        history_resp2 = client.get(f"/history/{session_id}")
        assert history_resp2.status_code == 200
        assert history_resp2.json()["total"] == 2

        # 两条记录的问题不同
        questions = [r["question"] for r in history_resp2.json()["records"]]
        assert "中芯国际" in questions[0]
        assert "产能利用率" in questions[1]

    def test_health_check_during_flow(self, client):
        """完整流程中健康检查始终可用"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
