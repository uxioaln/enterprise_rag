# 集成测试：POST /upload 接口（异步化 202 契约）与 GET /tasks/{task_id} 状态查询
import json
from unittest.mock import MagicMock, patch

import pandas as pd

from app.services import PipelineService


class TestUploadAPI:

    def test_upload_pdf_success(self, client, mock_pipeline, tmp_data_dir, make_mock_pdf_bytes, mock_celery_task):
        """成功上传 PDF：返回 202 受理响应（task_id + pending），文件保存到 pdf_reports，subset.csv 追加条目"""
        pdf_bytes = make_mock_pdf_bytes(256)
        file_name = "新研报.pdf"

        response = client.post(
            "/upload",
            files={"file": (file_name, pdf_bytes, "application/pdf")},
            data={"company_name": "中芯国际"},
        )

        # 异步化契约：202 + task_id，status 为 pending
        assert response.status_code == 202
        body = response.json()
        assert body["task_id"] == "test-task-id"
        assert body["status"] == "pending"

        # Celery 任务已提交，参数与受理信息一致
        mock_celery_task.delay.assert_called_once()
        args = mock_celery_task.delay.call_args.args
        assert args[0] == "中芯国际"  # company_name
        assert args[1] == file_name  # file_name
        assert args[2].startswith("stock_")  # sha1

        # 文件已保存到 pdf_reports
        saved_path = tmp_data_dir / "pdf_reports" / file_name
        assert saved_path.exists()
        assert saved_path.read_bytes() == pdf_bytes

        # subset.csv 已追加（受理前置步骤同步完成）
        df = pd.read_csv(tmp_data_dir / "subset.csv", encoding="utf-8")
        assert file_name in df["file_name"].values

    def test_upload_reject_non_pdf(self, client):
        """上传非 PDF 文件应返回 400"""
        response = client.post(
            "/upload",
            files={"file": ("report.txt", b"not a pdf", "text/plain")},
            data={"company_name": "中芯国际"},
        )
        assert response.status_code == 400
        assert "仅支持 PDF" in response.json()["detail"]

    def test_upload_reject_oversized(self, client, monkeypatch):
        """上传超大文件应返回 400"""
        # 把大小上限设为 1KB，方便测试
        monkeypatch.setattr("app.api.MAX_UPLOAD_SIZE", 1024)
        # 构造 2KB 的假 PDF
        pdf_bytes = b"%PDF-1.4\n" + b"\x00" * 2048 + b"\n%%EOF\n"

        response = client.post(
            "/upload",
            files={"file": ("big.pdf", pdf_bytes, "application/pdf")},
            data={"company_name": "中芯国际"},
        )
        assert response.status_code == 400
        assert "文件过大" in response.json()["detail"]

    def test_upload_missing_filename(self, client):
        """上传时空文件名应返回 422（请求体校验失败）"""
        response = client.post(
            "/upload",
            files={"file": ("", b"%PDF-1.4", "application/pdf")},
            data={"company_name": "中芯国际"},
        )
        assert response.status_code == 422

    def test_upload_celery_submit_failure_returns_500(self, client, mock_pipeline, mock_celery_task, make_mock_pdf_bytes):
        """Celery 任务提交失败时应返回 500"""
        mock_celery_task.delay.side_effect = RuntimeError("Redis broker 连接失败")

        response = client.post(
            "/upload",
            files={"file": ("fail.pdf", make_mock_pdf_bytes(64), "application/pdf")},
            data={"company_name": "中芯国际"},
        )
        assert response.status_code == 500
        assert "异步任务提交失败" in response.json()["detail"]

    def test_upload_register_failure_returns_500(self, client, mock_pipeline):
        """受理前置步骤（subset 登记）失败时应返回 500"""
        with patch.object(PipelineService, "register_pdf", side_effect=RuntimeError("subset 登记失败")):
            response = client.post(
                "/upload",
                files={"file": ("fail.pdf", b"%PDF-1.4 test", "application/pdf")},
                data={"company_name": "中芯国际"},
            )
        assert response.status_code == 500
        assert "文件登记失败" in response.json()["detail"]


class TestTaskStatusAPI:
    """GET /tasks/{task_id}：Celery 异步任务状态查询"""

    def test_task_status_success(self, client, monkeypatch):
        """SUCCESS 状态返回 success 与任务结果详情"""
        mock_result = MagicMock()
        mock_result.state = "SUCCESS"
        mock_result.result = {"file_name": "报告.pdf", "message": "入库完成"}
        monkeypatch.setattr("app.api.AsyncResult", MagicMock(return_value=mock_result))

        response = client.get("/tasks/some-task-id")
        assert response.status_code == 200
        body = response.json()
        assert body["task_id"] == "some-task-id"
        assert body["status"] == "success"
        assert body["detail"]["file_name"] == "报告.pdf"

    def test_task_status_failure(self, client, monkeypatch):
        """FAILURE 状态返回 failure 与异常信息"""
        mock_result = MagicMock()
        mock_result.state = "FAILURE"
        mock_result.result = RuntimeError("MinerU 解析超时")
        monkeypatch.setattr("app.api.AsyncResult", MagicMock(return_value=mock_result))

        response = client.get("/tasks/bad-task-id")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "failure"
        assert "MinerU 解析超时" in body["detail"]

    def test_task_status_pending(self, client, monkeypatch):
        """PENDING / RETRY 等中间状态统一返回 pending，detail 为空"""
        mock_result = MagicMock()
        mock_result.state = "RETRY"
        monkeypatch.setattr("app.api.AsyncResult", MagicMock(return_value=mock_result))

        response = client.get("/tasks/pending-task-id")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "pending"
        assert body["detail"] is None

    def test_task_status_nonexistent(self, client, monkeypatch):
        """不存在的 task_id（Celery 默认返回 PENDING）也返回 pending"""
        mock_result = MagicMock()
        mock_result.state = "PENDING"
        monkeypatch.setattr("app.api.AsyncResult", MagicMock(return_value=mock_result))

        response = client.get("/tasks/unknown-task-id")
        assert response.status_code == 200
        assert response.json()["status"] == "pending"
