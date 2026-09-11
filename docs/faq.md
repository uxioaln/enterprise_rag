# 常见问题（FAQ）

## 1. 报错"缺少 sha1 字段，无法保存 faiss 文件"

`subset.csv` 中缺少对应文件的 `sha1` 条目，或 `content_list.json` 文件名与 `subset.csv` 的 `file_name` 不匹配。确保 `content_list` 文件以 PDF 文件名（去扩展名）命名。

## 2. 相关页面显示为 0

旧版分块未保留页码。重新跑 `pipeline.chunk_reports()` 生成带真实页码的 chunked_reports 即可。

## 3. 切换嵌入模型后检索失败

嵌入模型维度不同（如 `text-embedding-3-small` 1536 维 vs `text-embedding-v4` 1024 维），FAISS 索引无法复用。删除旧索引后重新跑 `create_vector_dbs()`：

```bash
rm -f data/stock_data/databases/vector_dbs/*.faiss
```

## 4. MinerU 解析超时

MinerU 云端解析大文件可能较慢，可适当增加 `get_result` 的轮询等待时间。

## 5. AGICTO 调用报限流

调小 `parallel_requests`（建议为 1），或升级 AGICTO 平台额度。

## 6. /upload 报"文件登记或 OSS 上传失败"或"异步任务提交失败"

检查 `.env` 中是否正确配置了 OSS 凭证（`OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`）。"异步任务提交失败"通常意味着 Redis 未启动（Celery broker 连不上），先执行 `redis-server`。

## 7. /upload 返回 202 但任务一直是 pending

PDF 解析与入库由 Celery worker 执行，确认 worker 已在项目根目录启动且能连接同一 Redis：

```bash
celery -A app.celery_app:celery_app worker --loglevel=info --pool=threads
```

若 worker 日志显示任务最终失败（重试 3 次后），`GET /tasks/{task_id}` 会返回 `failure` 与具体错误。

## 8. 问答接口返回 HTTP 429

触发了 AGICTO 出口固定窗口限流（默认配额见 [docs/redis-cache.md](redis-cache.md)）。等待当前窗口结束再试，或在 `config.json` / `KB_RATE_LIMIT__LLM__LIMIT` 环境变量中按 AGICTO 配额调大 `limit`。

## 9. SSE 连接突然断开

服务端会检测客户端是否断开并取消模型流（原理见 [docs/sse-lifecycle.md](sse-lifecycle.md)），客户端主动关闭连接（如浏览器切换页面、fetch 被 abort）属于预期行为。

排查思路：

- 确认客户端是否主动关闭了 `EventSource` / `fetch` 流。
- 确认反向代理（Nginx 等）未设置过短的 `proxy_read_timeout`，建议 >= 120 秒（与 SSE 总超时一致）。
- 确认反向代理关闭了缓冲（响应头 `X-Accel-Buffering: no` 已由服务端下发）。

## 10. 模型服务超时

触发条件与超时阈值见 [docs/sse-lifecycle.md](sse-lifecycle.md)。重试建议：

- 限流导致超时：调小 Pipeline 的 `parallel_requests`（建议 1），或升级 AGICTO 额度，稍后重试。
- 大模型推理慢：确认 `answering_model` 配置，必要时换用更快的模型。
- 偶发网络抖动：直接重试即可。
