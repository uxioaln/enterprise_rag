# app/cache.py
"""异步 Redis 客户端与 embedding 缓存封装（redis.asyncio）。

职责：
1. 管理 Redis 连接：lifespan 启动时初始化全局连接池（init_redis），关闭时释放（close_redis）；
2. embedding 缓存：键格式 {embedding_key_prefix}{md5(query_text)}，SETEX 写入、GET 命中返回；
3. 降级策略：Redis 未启用或操作异常时不抛出异常，缓存视为未命中，业务直连原链路，不阻断服务。

同步调用方（如 src.api_requests.get_embeddings 所处的线程池上下文）通过
run_with_redis 桥接：在独立事件循环中使用临时连接执行协程，避免跨事件循环复用连接。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Awaitable, Callable, List, Optional, Sequence, Tuple, TypeVar

# redis 为可选依赖：本地 CLI 等未安装 redis 包的环境下，缓存与限流自动降级为直连，
# 不阻断导入与业务链路（服务端 Docker 环境安装 redis 后自动启用缓存/限流）
try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover - 取决于运行环境是否安装 redis
    aioredis = None

from app.config import get_config

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 全局异步 Redis 客户端（lifespan 启动时初始化，绑定主事件循环）
_global_redis: Optional[aioredis.Redis] = None
# Redis 是否可用（初始化成功为 True；初始化失败或未执行 lifespan 时为 False）
_enabled: bool = False

# 连接参数：短超时保证 Redis 故障时快速降级，不拖慢业务请求
_CONNECT_TIMEOUT = 2
_SOCKET_TIMEOUT = 2


def is_enabled() -> bool:
    """Redis 是否已启用（初始化成功且未被关闭）。"""
    return _enabled and _global_redis is not None


def get_global_redis() -> Optional[aioredis.Redis]:
    """获取全局 Redis 客户端（仅供 async 上下文使用，未启用时返回 None）。"""
    return _global_redis if _enabled else None


def build_key(text: str) -> str:
    """构建 embedding 缓存键：{prefix}{md5(text)}。

    前缀来自配置 embedding_key_prefix（如 emb:v4:），
    更换 embedding 模型时修改前缀即可隔离新旧维度的向量。
    """
    prefix = get_config().redis.embedding_key_prefix
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}"


def _create_redis() -> aioredis.Redis:
    """按当前配置创建新的异步 Redis 客户端（短超时，快速失败）。"""
    cfg = get_config().redis
    return aioredis.Redis.from_url(
        cfg.url,
        decode_responses=True,
        socket_connect_timeout=_CONNECT_TIMEOUT,
        socket_timeout=_SOCKET_TIMEOUT,
    )


# --------------------------------------------------------------------------- #
# 生命周期：lifespan 启动初始化 / 关闭释放
# --------------------------------------------------------------------------- #
async def init_redis() -> bool:
    """初始化全局 Redis 连接池并 PING 验证。成功返回 True，失败降级返回 False。"""
    global _global_redis, _enabled
    try:
        client = _create_redis()
        await client.ping()
        _global_redis = client
        _enabled = True
        logger.info("Redis 连接成功: %s", get_config().redis.url)
        return True
    except Exception as exc:
        # 初始化失败不阻断服务：缓存与限流自动降级为直连原链路
        logger.warning("Redis 初始化失败，缓存与限流将降级为直连模式: %s", exc)
        _global_redis = None
        _enabled = False
        return False


async def close_redis() -> None:
    """释放全局 Redis 连接，归还连接池资源。幂等，多次调用安全。"""
    global _global_redis, _enabled
    _enabled = False
    if _global_redis is not None:
        try:
            await _global_redis.aclose()
        except Exception:
            logger.warning("Redis 连接关闭异常（忽略）", exc_info=True)
        _global_redis = None
        logger.info("Redis 连接已释放")


# --------------------------------------------------------------------------- #
# embedding 缓存：async 原语（全局连接，主事件循环内 await 使用）
# --------------------------------------------------------------------------- #
async def get_cached_embedding(text: str, redis: Optional[aioredis.Redis] = None) -> Optional[List[float]]:
    """读取单条 embedding 缓存。命中返回向量，未命中或异常返回 None（视为缓存未命中）。"""
    client = redis if redis is not None else get_global_redis()
    if client is None:
        return None
    try:
        raw = await client.get(build_key(text))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        # Redis 异常降级：视为未命中，不阻断服务
        logger.warning("embedding 缓存 GET 失败，降级为未命中: %s", exc)
        return None


async def get_cached_embeddings(
    texts: Sequence[str], redis: Optional[aioredis.Redis] = None
) -> List[Optional[List[float]]]:
    """批量读取 embedding 缓存（pipeline 一次往返），返回与 texts 等长的结果列表。"""
    client = redis if redis is not None else get_global_redis()
    results: List[Optional[List[float]]] = [None] * len(texts)
    if client is None or not texts:
        return results
    try:
        async with client.pipeline(transaction=False) as pipe:
            for text in texts:
                pipe.get(build_key(text))
            raws = await pipe.execute()
        for i, raw in enumerate(raws):
            if raw is not None:
                results[i] = json.loads(raw)
        return results
    except Exception as exc:
        logger.warning("embedding 缓存批量 GET 失败，降级为全量未命中: %s", exc)
        return [None] * len(texts)


async def set_cached_embeddings(
    pairs: Sequence[Tuple[str, Sequence[float]]],
    redis: Optional[aioredis.Redis] = None,
    ttl: Optional[int] = None,
) -> None:
    """批量写入 embedding 缓存（pipeline SETEX）。TTL 默认取配置 embedding_ttl。

    写入失败仅记录日志，不影响业务结果。
    """
    client = redis if redis is not None else get_global_redis()
    if client is None or not pairs:
        return
    if ttl is None:
        ttl = get_config().redis.embedding_ttl
    try:
        async with client.pipeline(transaction=False) as pipe:
            for text, vector in pairs:
                pipe.set(build_key(text), json.dumps(vector, separators=(",", ":")), ex=ttl)
            await pipe.execute()
    except Exception as exc:
        # 缓存写入失败不影响本次请求结果
        logger.warning("embedding 缓存 SETEX 写入失败（忽略）: %s", exc)


# --------------------------------------------------------------------------- #
# 同步桥接：供线程池 / CLI 等无事件循环上下文使用
# --------------------------------------------------------------------------- #
# run_with_redis 的降级哨兵：表示连接 Redis 失败，调用方应直连原链路
DEGRADED = object()


def run_with_redis(coro_fn: Callable[[aioredis.Redis], Awaitable[T]]) -> T:
    """在独立事件循环中以临时 Redis 连接执行 coro_fn(redis)，供同步代码调用。

    - 连接失败：返回 DEGRADED 哨兵（不抛异常），调用方据此降级为直连；
    - coro_fn 内部的业务异常（如 RateLimitExceeded、API 错误）原样向上抛出；
    - 当前线程已存在运行中的事件循环时抛出 RuntimeError，调用方应改用 await 原语。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 无运行中的事件循环（线程池 / CLI 上下文），可以安全桥接
        pass
    else:
        raise RuntimeError(
            "run_with_redis 不能在事件循环内调用，请在 async 上下文中直接 await cache 原语"
        )

    async def _runner():
        try:
            client = _create_redis()
            await client.ping()
        except Exception as exc:
            logger.warning("Redis 连接失败，降级为直连: %s", exc)
            return DEGRADED
        try:
            return await coro_fn(client)
        finally:
            try:
                await client.aclose()
            except Exception:
                pass

    return asyncio.run(_runner())
