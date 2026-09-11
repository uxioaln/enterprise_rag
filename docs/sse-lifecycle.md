# SSE 连接生命周期管理

`process_chat` 采用 **生产者 / 消费者 + 队列** 模式实现完整的连接生命周期管理（见 [app/services.py](../app/services.py)）。

## 核心机制

- **客户端断开检测与取消**：路由层将 `Request` 对象注入 `process_chat`；每推送 4 个 delta chunk 以及 keep-alive 超时时检测 `await request.is_disconnected()`，一旦发现客户端断开立即取消生产者任务，触发 `async with stream:` 退出关闭 openai 流，避免断开后继续消耗模型 token。
- **keep-alive 保活**：若 30 秒内无数据产生，下发一行 SSE 注释帧 `: keep-alive\n\n` 维持连接，防止客户端 / 反向代理因空闲超时断开。
- **总超时**：单次 SSE 响应总超时 120 秒（`_TOTAL_TIMEOUT`），超时后主动推送 `timeout` error 并关闭连接。
- **分层异常处理**：流式生成器内按异常类型推送结构化 `error` 事件（事件类型表见 [docs/api.md](api.md#4-sse-事件流)）。
- **资源清理**：消费者退出时取消生产者任务，由 `async with stream:` 归还 HTTP 连接；应用关闭（lifespan shutdown）时调用 `PipelineService.aclose()` 关闭 `AsyncOpenAI` 客户端，并通过 `gc.collect()` 释放 FAISS / BM25 索引等全局资源。

## 关键代码示意

（见 [app/services.py](../app/services.py)）

```python
stream = await client.chat.completions.create(
    model=answering_model, messages=[...], stream=True,
)
# async with 包裹：无论正常退出、异常或取消，都会关闭流并归还 HTTP 连接
async with stream:
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content  # 直接访问 delta，不解析 SSE 字符串
        if delta:
            yield delta
```

## 排查提示

- 客户端主动关闭连接（如浏览器切换页面、fetch 被 abort）属于预期行为，服务端会检测断开并取消模型流。
- 确认反向代理（Nginx 等）未设置过短的 `proxy_read_timeout`，建议 >= 120 秒（与 SSE 总超时一致）。
- 确认反向代理关闭了缓冲（响应头 `X-Accel-Buffering: no` 已由服务端下发）。
