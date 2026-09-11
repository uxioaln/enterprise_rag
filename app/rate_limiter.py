# app/rate_limiter.py
"""AGICTO 出口固定窗口限流器（基于 Redis）。

按业务桶区分配额：
- rl:agicto:embed -> embedding 接口（配置 rate_limit.embedding）
- rl:agicto:llm   -> 问答 / 重排接口（配置 rate_limit.llm）

核心算法（固定窗口，约 10 行）：
    current_window = int(time.time()) // window
    redis_key = f"{key}:{current_window}"
    count = redis.incr(redis_key)
    if count == 1:
        redis.expire(redis_key, window)
    return count <= limit

Redis 未启用或操作异常时放行（fail-open），不阻断服务。
"""
from __future__ import annotations

import logging
import time
from typing import Optional

# redis 为可选依赖：未安装 redis 包的环境（如本地 CLI）限流自动放行，不阻断导入
try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover - 取决于运行环境是否安装 redis
    aioredis = None

from app import cache
from app.config import RateLimitRule, get_config

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """限流触发异常：由 ErrorHandlingMiddleware 捕获并映射为 HTTP 429。"""


def _rule_for(bucket: str) -> RateLimitRule:
    """按业务桶名读取限流规则，未知桶名抛出 ValueError。"""
    rate_limit_cfg = get_config().rate_limit
    rule = getattr(rate_limit_cfg, bucket, None)
    if rule is None:
        raise ValueError(f"未知的限流业务桶: {bucket}，可用: embedding / llm")
    return rule


async def check_rate_limit(bucket: str, redis: Optional[aioredis.Redis] = None) -> None:
    """固定窗口限流检查：超限抛出 RateLimitExceeded，Redis 异常时放行。

    参数 bucket 为业务桶名（"embedding" / "llm"），Redis 键为 rl:agicto:{bucket}:{window}。
    可通过 redis 参数注入连接（同步桥接场景），缺省使用全局连接。
    """
    client = redis if redis is not None else cache.get_global_redis()
    if client is None:
        # Redis 未启用：放行（服务降级为无限流）
        return
    rule = _rule_for(bucket)

    # ---- 固定窗口核心算法 ----
    current_window = int(time.time()) // rule.window
    redis_key = f"rl:agicto:{bucket}:{current_window}"
    try:
        count = await client.incr(redis_key)
        if count == 1:
            await client.expire(redis_key, rule.window)
    except Exception as exc:
        logger.warning("限流检查 Redis 异常，本次放行: %s", exc)
        return
    if count > rule.limit:
        raise RateLimitExceeded(
            f"请求过于频繁，{bucket} 接口超过 {rule.limit} 次/{rule.window}s 限制，请稍后再试"
        )


def check_rate_limit_sync(bucket: str) -> None:
    """同步上下文的限流检查入口（线程池 / CLI）。

    Redis 未启用时直接放行；启用时通过 run_with_redis 桥接执行。
    RateLimitExceeded 会原样向上抛出，Redis 连接失败时放行。
    """
    if not cache.is_enabled():
        return
    cache.run_with_redis(lambda client: check_rate_limit(bucket, redis=client))
