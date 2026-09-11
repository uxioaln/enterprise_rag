# 集成测试：GET /health 健康检查接口


class TestHealthAPI:

    def test_health_check(self, client):
        """健康检查应返回 200 且 status 为 ok"""
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"

    def test_health_content_type(self, client):
        """健康检查应返回 JSON 格式"""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")
