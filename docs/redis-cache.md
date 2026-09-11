# Redis Embedding 缓存与限流

Redis 在本项目中承担三个角色：embedding 缓存、AGICTO 出口固定窗口限流计数、Celery broker/backend。实现分别位于 [app/cache.py](../app/cache.py) 与 [app/rate_limiter.py](../app/rate_limiter.py)。

## 1. Embedding 缓存

在调用 AGICTO `text-embedding-v4` 前先查 Redis：

- 键格式：`{embedding_key_prefix}{md5(text)}`，默认前缀 `emb:v4:`，即 `emb:v4:{md5(query_text)}`
- TTL：默认 3600 秒，可通过 `redis.embedding_ttl` 配置
- 命中直接返回向量（毫秒级）；未命中才真实调用，并在拿到结果后 SETEX 写回
- 更换 embedding 模型时修改 `embedding_key_prefix`（如 `emb:v5:`）即可隔离新旧维度向量

## 2. 固定窗口限流

在 AGICTO 出口处按业务分桶限流：

| 限流桶 | Redis 键 | 默认配额 |
|---|---|---|
| embedding | `rl:agicto:embed` | 60 秒 / 100 次 |
| 问答 / 重排（LLM） | `rl:agicto:llm` | 60 秒 / 30 次 |

- 窗口计数用 Redis `INCR` + `EXPIRE` 实现。
- 触发限流抛出 `RateLimitExceeded`，由 `ErrorHandlingMiddleware` 统一返回 HTTP 429：

```json
{"error": {"type": "rate_limit", "message": "请求过于频繁，请稍后再试", "request_id": "..."}}
```

## 3. 降级策略

Redis 初始化失败或运行中异常时，缓存视为未命中、限流直接放行（记录 warning 日志），业务直连 AGICTO，**不阻断服务**。

## 4. 缓存命中不消耗限流配额

限流检查位于缓存未命中后的真实调用之前，命中缓存的请求不计数。

## 5. 相关配置

- Redis 连接：默认 `redis://localhost:6379/0`，可通过 `KB_REDIS__URL` 环境变量覆盖（嵌套用双下划线，详见 [docs/configuration.md](configuration.md)）。
- Docker Compose 部署时 `KB_REDIS__URL` 已由 compose 文件覆盖为 `redis://redis:6379/0`，无需在 `.env` 中重复设置。
- 限流配额可在 `config.json` 的 `rate_limit` 段或 `KB_RATE_LIMIT__LLM__LIMIT` 等环境变量中调整。
