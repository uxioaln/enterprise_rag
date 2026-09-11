# 配置中心与热重载

应用配置集中在 [app/config.py](../app/config.py)，由 `AppConfig` 聚合 `ModelConfig` / `RetrievalConfig` / `PipelineConfig` / `RedisConfig` / `RateLimitConfig` 五个 `dataclass(frozen=True, slots=True)` 子配置。配置分层优先级：

```
dataclass 默认值  <  config.json  <  KB_ 前缀环境变量
```

- `load_config()`（异步）：在 `lifespan` 启动时加载 [config.json](../config.json)，并合并 `KB_` 前缀环境变量（嵌套用双下划线，如 `KB_MODEL__LLM_MODEL`），自动做布尔 / 整型 / 浮点类型推断。
- `maybe_reload_config()`：由 `lifespan` 启动的后台协程 `_config_watcher` 每 10 秒检测 `config.json` 的 mtime，发现变更即自动热重载，**无需重启服务**。
- `get_config()`：同步全局访问接口，供任意上下文读取当前配置。

## KB_ 前缀环境变量

嵌套配置项用双下划线连接，常用示例：

- `KB_MODEL__LLM_MODEL`
- `KB_REDIS__EMBEDDING_TTL`
- `KB_RATE_LIMIT__LLM__LIMIT`
- `KB_RETRY_LOOP__ENABLE=true`

Docker 部署时，`./config.json` 单独挂载到 `/app/config.json`，宿主机修改后自动热重载，无需重启容器（原理即上文 mtime 检测）。

## 应用层环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `STORAGE_BACKEND` | `sqlite` | 会话存储后端：`sqlite`（默认）/ `memory`，详见下文"对话历史持久化" |
| `KB_DB_PATH` | `data/chat_history.db` | SQLite 数据库文件路径 |
| `ALLOWED_ORIGINS` | 开发环境前端来源 | CORS 允许来源（逗号分隔），详见 [docs/api.md](api.md#7-cors-跨域配置) |
| `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` / `OSS_BUCKET_NAME` / `OSS_ENDPOINT` | 无 / `vl-image` / `https://oss-cn-shanghai.aliyuncs.com` | 阿里云 OSS 凭证，用于 `/upload` 上传 PDF 供 MinerU 云端解析读取 |

## 对话历史持久化（SQLite / 内存双后端）

会话与问答历史通过 [app/storage.py](../app/storage.py) 的统一 `SessionStorage` 异步协议抽象，由 `STORAGE_BACKEND` 环境变量选择实现（[app/main.py](../app/main.py) 的 `lifespan` 启动时调用 `get_storage()` 装配）：

| 后端 | 取值 | 说明 |
|---|---|---|
| SQLite（默认，推荐生产） | `STORAGE_BACKEND=sqlite` | 基于 [app/db.py](../app/db.py) 的 `SQLiteSessionStorage`，使用 `aiosqlite` 异步 IO，WAL 模式优化并发，外键级联保证数据完整。完整保留 `step_by_step_analysis` / `reasoning_summary` / `relevant_pages` / `references` 等字段（JSON 序列化存储）。 |
| 内存（开发/测试） | `STORAGE_BACKEND=memory` | 基于 `InMemorySessionStorage`，零依赖、纯内存，与 SQLite 实现同一异步协议，可无缝替换。 |

SQLite 表结构（[app/db.py](../app/db.py) 的 `init_db` 幂等建表）：

- `sessions(session_id, created_at, updated_at, metadata)`：会话元数据
- `chat_history(id, session_id, question, answer, step_by_step_analysis, reasoning_summary, relevant_pages, references_json, elapsed_seconds, created_at)`：问答历史，按 `session_id` 索引、外键级联删除
- 数据库文件默认 `data/chat_history.db`，可通过 `KB_DB_PATH` 环境变量覆盖

> `InMemoryStorage`（早期同步实现）保留用于既有单元测试与向后兼容，异步服务层统一使用 `InMemorySessionStorage` / `SQLiteSessionStorage`。
