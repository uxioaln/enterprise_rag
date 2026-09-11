# API 详细说明

FastAPI 服务（`app/main.py`）的接口细节、SSE 事件流、调用示例与全局中间件说明。接口概览见 [README](../README.md#api-概览)。

## 1. 接口总表

| 接口 | 方法 | 说明 |
|---|---|---|
| `/upload` | POST | 上传 PDF 受理入库：保存文件 -> 登记 subset -> 提交 Celery 异步任务，**立即返回 202**（`task_id` + `status=pending`）；解析/分块/向量化由 worker 后台执行 |
| `/tasks/{task_id}` | GET | 查询 PDF 入库异步任务状态，返回 `pending` / `success` / `failure` 与结果详情 |
| `/chat` | POST | 核心问答接口，默认 SSE 流式逐句返回；`?stream=false` 时返回标准 JSON（`ChatResponse` 模型） |
| `/history/{session_id}` | GET | 获取指定会话的历史问答记录，不存在时返回 404 |
| `/health` | GET | 健康检查，返回 `{"status": "ok"}` |

## 2. PDF 上传异步化（/upload + /tasks）

`/upload` 已从同步阻塞改为 Celery 异步任务（[app/api.py](../app/api.py) / [app/tasks.py](../app/tasks.py)）：

1. API 受理：保存 PDF 到 `pdf_reports/` -> 登记 `subset.csv`（取得 sha1）-> 提交 Celery 任务 `parse_and_index_pdf` -> **立即返回 202**：
   ```json
   {"task_id": "a1b2c3d4-...", "status": "pending"}
   ```
2. 后台执行（Celery worker）：MinerU 云端解析（签名 URL 直传本地 PDF 文件，无需 OSS）-> 真实页码分块 -> 增量向量化入库（复用既有 `src/pdf_mineru.py` / `text_splitter.py` / `ingestion.py` 逻辑）；失败自动重试（`countdown=10`，`max_retries=3`）。
3. 查询进度：`GET /tasks/{task_id}` 返回 `pending`（排队/执行中）/ `success` / `failure` 及结果详情。

并发安全：多个入库任务同时向 FAISS 索引写入时，由 [src/ingestion.py](../src/ingestion.py) 的 `VectorDBIngestor.add_chunks_to_index_atomic`（进程内互斥锁 + 临时文件原子替换落盘）保证索引不被写坏。

## 3. SSE 流式实现说明

`/chat` 接口的 SSE 流式响应基于 `openai` 库的原生异步流式迭代器（`AsyncStream`）实现，**不手动拼接或解析 `"data: {...}"` SSE 字符串**，也**不使用 `str(chunk).split(...)` / `json.loads` 处理原始 SSE 行**。核心流程（见 [app/services.py](../app/services.py) 的 `PipelineService.stream_final_answer` 与 `process_chat`）：

1. 推理阶段：调用 `Pipeline.answer_single_question`（放入线程池，避免阻塞事件循环）拿到结构化答案，将 `step_by_step_analysis` / `reasoning_summary` 映射为 `reasoning` 事件，推理就绪即推送，不等最终答案生成完再一次性下发。推理受总超时（120 秒）约束，超时推送 `timeout` error。
2. 答案阶段：直接使用 `AsyncOpenAI` 客户端调用 `chat.completions.create(..., stream=True)` 拿到 `AsyncStream` 迭代器，遍历时**直接访问 `chunk.choices[0].delta.content`** 作为增量文本，逐 chunk 通过 `delta` 事件下发；使用 `async with stream:` 包裹流迭代器，确保底层 HTTP 连接被归还到连接池，即使发生异常或外部取消也能正确关闭流。

SSE 连接生命周期管理（断开检测、keep-alive、超时、资源清理）详见 [docs/sse-lifecycle.md](sse-lifecycle.md)。

## 4. SSE 事件流

`/chat`（默认 `stream=true`）按顺序下发以下 SSE 事件，每个事件的 `data` 字段均为合法 JSON 字符串。此外，长时间无数据时会下发 SSE 注释帧 `: keep-alive\n\n`（非事件，仅维持连接）：

| 事件 | 说明 | data 结构 |
|---|---|---|
| `start` | 连接建立，下发会话与问题信息 | `{"session_id": "...", "question": "..."}` |
| `retry_start` | 重试循环概览（仅启用重试且发生多轮时推送） | `{"total_attempts": 3, "forced_exit": false}` |
| `retry_reasoning` | 每轮重试的查询、置信度与反思（可多次推送） | `{"attempt": 1, "query": "...", "confidence": {...}, "critique": "...", "strategy": "expand", "rationale": "..."}` |
| `retry_success` | 最终采纳的置信度与 forced_exit 状态 | `{"final_confidence": {...}, "forced_exit": false}` |
| `reasoning` | 结构化推理（来自 Pipeline 推理结果，推理就绪即推送） | `{"step_by_step_analysis": "...", "reasoning_summary": "..."}` |
| `delta` | 最终答案的逐 chunk 增量内容（来自 openai 原生流 `chunk.choices[0].delta.content`，可多次推送） | `{"content": "..."}` |
| `done` | 最终完整结果，含相关页面与引用，并写入会话历史（含 confidence / retry_metadata / forced_exit） | `{"session_id": "...", "question": "...", "answer": "...", "relevant_pages": [...], "references": [...], "elapsed_seconds": 12.34, "confidence": {...}, "retry_metadata": {...}, "forced_exit": false}` |
| `error` | 推理或流式调用异常时推送 | `{"type": "...", "message": "..."}` |

`error` 事件字段说明：

| `type` | 触发条件 | `message` 示例 |
|---|---|---|
| `rate_limit` | AGICTO 平台返回限流（`openai.RateLimitError`） | `当前请求过于频繁，请稍后再试` |
| `timeout` | 模型响应超时 / 连接错误（`openai.APITimeoutError` / `APIConnectionError`）或总超时 120 秒触发 | `模型服务响应超时，请重试` |
| `business` | 业务异常（检索不到文档、FAISS 索引未找到等） | `推理失败: 未找到 FAISS 索引文件` |
| `internal` | 未预料异常（堆栈已记录到服务端日志，不向客户端暴露原始信息） | `系统内部错误` |

事件顺序：`start` -> `reasoning` -> `delta`（多次）-> `done`；任一阶段异常时推送 `error` 事件并终止该次流。客户端断开或总超时时直接关闭连接，不再推送 `done`。

`error` 事件示例：

```
event: error
data: {"type": "rate_limit", "message": "当前请求过于频繁，请稍后再试"}

```

keep-alive 注释帧示例（无 `event` / `data`，仅以 `:` 开头的注释）：

```
: keep-alive

```

## 5. 调用示例

```bash
# 1) 上传 PDF（异步受理：需指定 company_name 以便后续问答路由；需先启动 Redis 与 Celery worker）
curl -X POST http://localhost:8000/upload \
  -F "file=@研报.pdf" \
  -F "company_name=中芯国际"

# 响应（202 Accepted，解析与入库由 Celery worker 后台执行）：
# {"task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "status": "pending"}

# 1.1) 查询入库任务进度
curl http://localhost:8000/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890

# 状态查询响应示例：
# pending（排队/执行中）:  {"task_id": "a1b2...", "status": "pending", "detail": null}
# success（入库完成）:     {"task_id": "a1b2...", "status": "success", "detail": {"file_name": "研报.pdf", "message": "MinerU 解析、分块与向量化入库完成", ...}}
# failure（最终失败）:     {"task_id": "a1b2...", "status": "failure", "detail": "MinerU 解析超时..."}

# 2) 流式问答（SSE）：-N 关闭缓冲，逐 event 实时接收
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess-01", "question": "\"中芯国际\"在晶圆制造行业中的地位如何？"}'

# SSE 响应示例（每个事件含 event 行 + data(JSON) 行 + 空行）：
# event: start
# data: {"session_id": "sess-01", "question": "\"中芯国际\"在晶圆制造行业中的地位如何？"}
#
# event: reasoning
# data: {"step_by_step_analysis": "...", "reasoning_summary": "..."}
#
# event: delta
# data: {"content": "中芯国际是全球领先的集成电路晶圆代工企业。"}
#
# event: delta
# data: {"content": "其产能利用率在2024年显著提升。"}
#
# event: done
# data: {"session_id": "sess-01", "question": "...", "answer": "...", "relevant_pages": [...], "references": [...], "elapsed_seconds": 8.21}
#
```

```bash
# 2.1) 422 请求体校验失败示例：question 未用英文双引号包裹公司名
#      注意：校验失败在建立 SSE 连接前直接返回 HTTP 422，不会进入 SSE 流
curl -i -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess-01", "question": "中芯国际在晶圆制造行业中的地位如何"}'

# HTTP/1.1 422 Unprocessable Entity
# content-type: application/json
#
# {
#   "detail": [
#     {
#       "type": "value_error",
#       "loc": ["body", "question"],
#       "msg": "Value error, 问题中必须使用英文双引号包裹公司名",
#       "input": "中芯国际在晶圆制造行业中的地位如何"
#     }
#   ]
# }
```

```bash
# 3) 非流式问答（标准 JSON，返回 ChatResponse 模型）
curl -X POST "http://localhost:8000/chat?stream=false" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "question": "\"中芯国际\"的产能利用率如何？"}'

# 4) 查询会话历史
curl http://localhost:8000/history/s1
```

## 6. 全局中间件

[app/middleware.py](../app/middleware.py) 提供两个全局中间件，在 [app/main.py](../app/main.py) 的 `create_app()` 中按以下顺序挂载（外层到内层）：

| 中间件 | 职责 |
|---|---|
| `ErrorHandlingMiddleware`（最外层） | 捕获所有未处理异常，记录完整堆栈到服务端日志，按异常类型映射 HTTP 状态码（含 `RateLimitExceeded` -> 429），返回标准 `{"error": {"type", "message", "request_id"}}` JSON，**不向客户端泄露堆栈信息** |
| `RequestLoggingMiddleware` | 为每个请求生成唯一 `request_id`（注入 `request.state` 与响应头 `X-Request-ID`），记录 method / path / query / status / duration_ms / client / user_agent，按状态码选择日志级别，输出结构化 JSON 日志（跳过 `/health` / `/docs` / `/openapi.json` / `/redoc`） |
| `CORSMiddleware`（最内层，路由前） | 全局跨域处理 |

## 7. CORS 跨域配置

应用在 [app/main.py](../app/main.py) 的 `create_app()` 中通过 `CORSMiddleware` 全局挂载 CORS，保证前端联调（含 SSE 端点 `/chat` 的预检 OPTIONS 正常返回 200）：

- 允许方法：`GET` / `POST` / `OPTIONS`
- 允许 Headers：`Content-Type` / `Authorization`
- 允许来源：优先读环境变量 `ALLOWED_ORIGINS`（逗号分隔）；未设置时默认允许开发环境前端 `http://localhost:3000`、`http://localhost:5173`、`http://localhost:8080`
- `allow_credentials=True`（与具体来源配合）；注意 CORS 规范禁止 `allow_credentials=True` 与通配符 `*` 同时使用，因此当 `ALLOWED_ORIGINS=*` 时自动关闭 `allow_credentials` 以避免运行时报错
