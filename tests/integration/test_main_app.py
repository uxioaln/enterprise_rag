# 集成测试：app/main.py 的 create_app 与 lifespan
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestCreateApp:

    def test_create_app_returns_fastapi_instance(self):
        """create_app() 应返回 FastAPI 实例"""
        from app.main import create_app

        application = create_app()
        assert isinstance(application, FastAPI)

    def test_create_app_registers_core_routes(self, monkeypatch, tmp_data_dir):
        """create_app() 返回的应用应注册全部核心路由（通过 OpenAPI schema 验证）"""
        from app.main import create_app

        # Mock Pipeline 避免真实初始化
        monkeypatch.setattr("app.main.Pipeline", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr("app.main.here", lambda: tmp_data_dir.parent)

        application = create_app()
        with TestClient(application) as c:
            schema = c.get("/openapi.json").json()
            paths = set(schema["paths"].keys())
            assert "/upload" in paths
            assert "/chat" in paths
            assert "/history/{session_id}" in paths
            assert "/health" in paths

    def test_create_app_metadata(self):
        """create_app() 应正确设置应用元数据"""
        from app.main import create_app

        application = create_app()
        assert application.title == "企业知识库 RAG 问答服务"
        assert application.version == "1.0.0"

    def test_docs_endpoint_returns_200(self, client):
        """FastAPI 自动文档端点 /docs 应返回 200"""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_endpoint_returns_200(self, client):
        """OpenAPI schema 端点 /openapi.json 应返回 200"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/health" in schema["paths"]


class TestLifespan:
    """测试 lifespan 事件：初始化与清理"""

    def test_lifespan_initializes_app_state(self, monkeypatch, tmp_data_dir):
        """lifespan 启动时应初始化 pipeline_service 和 storage"""
        from app.main import create_app

        # Mock Pipeline 构造函数，避免真实初始化（需 AGICTO_API_KEY 等）
        mock_pipeline_instance = MagicMock()
        mock_pipeline_cls = MagicMock(return_value=mock_pipeline_instance)
        monkeypatch.setattr("app.main.Pipeline", mock_pipeline_cls)

        # Mock here() 返回临时目录，避免依赖项目真实路径
        monkeypatch.setattr("app.main.here", lambda: tmp_data_dir.parent)

        application = create_app()
        with TestClient(application) as c:
            # startup 已执行，app.state 应已初始化
            assert hasattr(application.state, "pipeline_service")
            assert hasattr(application.state, "storage")
            # 健康检查可用
            assert c.get("/health").json()["status"] == "ok"

        # shutdown 后应清理
        assert application.state.pipeline_service is None
        assert application.state.storage is None

    def test_lifespan_uses_max_config(self, monkeypatch, tmp_data_dir):
        """lifespan 应使用 max_config 初始化 Pipeline"""
        from app.main import create_app
        from src.pipeline import max_config

        mock_pipeline_cls = MagicMock(return_value=MagicMock())
        monkeypatch.setattr("app.main.Pipeline", mock_pipeline_cls)
        monkeypatch.setattr("app.main.here", lambda: tmp_data_dir.parent)

        application = create_app()
        with TestClient(application):
            pass  # 触发 startup + shutdown

        # Pipeline 应被调用，且 run_config=max_config
        mock_pipeline_cls.assert_called_once()
        _, kwargs = mock_pipeline_cls.call_args
        assert kwargs.get("run_config") is max_config


class TestRun:
    """测试 uvicorn 启动入口函数"""

    def test_run_calls_uvicorn(self, monkeypatch):
        """run() 应调用 uvicorn.run 并传入正确的参数"""
        from app.main import run
        import sys

        mock_uvicorn = MagicMock()
        monkeypatch.setitem(sys.modules, "uvicorn", mock_uvicorn)

        run()
        mock_uvicorn.run.assert_called_once_with("app.main:app", host="0.0.0.0", port=8000)
