# 测试指南

项目包含完整的分层自动化测试（单元测试 / 集成测试 / E2E 测试），所有外部依赖（MinerU、AGICTO、FAISS、OSS）均被 Mock，确保测试快速、稳定且独立。

## 安装测试依赖

```bash
pip install -r requirements-test.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行全部测试

```bash
python -m pytest tests/ -v
```

## 生成覆盖率报告

```bash
python -m pytest tests/ --cov=app --cov-report=term-missing
```

## 覆盖率明细

当前共 55 个测试用例全部通过。app 模块整体覆盖率 67%，其中 `api.py` 96%、`schemas.py` 94%、`celery_app.py` 85%、`main.py` 85%；`/upload` 异步 202 契约、`/tasks/{task_id}` 状态查询（pending / success / failure / 不存在）均已覆盖（Celery 任务提交与 Redis broker 全部 Mock，测试无需真实 Redis）。`cache.py` / `rate_limiter.py` / `tasks.py` 的 Redis 真实交互与重试路径未纳入默认测试（避免外部依赖），经 lifespan 降级冒烟验证。`services.py` 中 SSE 连接生命周期管理新增的限流 / 超时 / 内部异常、keep-alive、客户端断开检测等分支因依赖真实网络错误与断连场景被 Mock，未计入覆盖。

## 测试目录结构

- [tests/conftest.py](../tests/conftest.py)：全局 fixtures（mock_pipeline、TestClient、临时数据目录、异步内存存储）
- [tests/unit/](../tests/unit/)：单元测试（storage / upload_service / chat_service）
- [tests/integration/](../tests/integration/)：集成测试（4 个 API + main_app）
- [tests/e2e/](../tests/e2e/)：端到端测试（完整用户流程）
