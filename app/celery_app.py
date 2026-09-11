# app/celery_app.py
"""Celery 应用实例：Redis 作为 broker 与 backend。

worker 独立启动命令（与项目结构兼容，需在项目根目录执行）：
    celery -A app.celery_app:celery_app worker --loglevel=info --pool=threads --concurrency=4

使用 threads 池原因：任务内含 FAISS 索引更新与 MinerU 阻塞 HTTP 调用，
线程池共享进程内存，配合 src.ingestion.VectorDBIngestor 的类级锁可安全串行化索引更新。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from celery import Celery

logger = logging.getLogger(__name__)

# Redis URL 解析优先级：KB_REDIS__URL 环境变量 -> config.json redis.url -> 默认值
# （Celery 模块可能在 lifespan 加载配置前被 worker 导入，因此这里同步读取一次）
_DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def _resolve_redis_url() -> str:
    """解析 Celery broker/backend 使用的 Redis URL。"""
    env_url = os.getenv("KB_REDIS__URL", "").strip()
    if env_url:
        return env_url
    config_path = Path(__file__).resolve().parent.parent / "config.json"
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            url = str(data.get("redis", {}).get("url", "")).strip()
            if url:
                return url
    except Exception as exc:
        logger.warning("读取 config.json 中 redis.url 失败，使用默认值: %s", exc)
    return _DEFAULT_REDIS_URL


_REDIS_URL = _resolve_redis_url()

# Celery 应用实例：任务定义在 app.tasks（include 确保 worker 启动时自动注册）
celery_app = Celery(
    "knowledge_base",
    broker=_REDIS_URL,
    backend=_REDIS_URL,
    include=["app.tasks"],
)

# 任务序列化与结果配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=3600,
    timezone="Asia/Shanghai",
    enable_utc=False,
    # 任务开始执行即更新状态（便于 /tasks/{task_id} 区分排队与执行中）
    task_track_started=True,
    # broker 连接失败时启动阶段自动重试，避免 worker 因 Redis 瞬断退出
    broker_connection_retry_on_startup=True,
    # 单个任务硬超时：MinerU 解析含轮询等待，放宽到 30 分钟
    task_soft_time_limit=1740,
    task_time_limit=1800,
)
