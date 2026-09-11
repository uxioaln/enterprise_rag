# 低置信度自动查询改写重试

问答后自动评估答案质量，低于阈值时自动改写查询并重试，仅改编排层，不改动现有 retrieve / rerank / generate 内部逻辑。由 [src/pipeline.py](../src/pipeline.py) 的 `Pipeline._answer_with_retry_loop` 在 `RunConfig.enable_retry_loop=True` 时激活。

## 循环逻辑

（[src/pipeline.py](../src/pipeline.py) / [src/retrieval_evaluator.py](../src/retrieval_evaluator.py) / [src/query_rewriter.py](../src/query_rewriter.py)）：

```
for attempt in range(max_retries + 1):
    docs = retrieve(query)          # 复用现有 _retrieve_contexts
    answer = generate(query, docs)  # 复用现有 _generate_answer
    confidence = evaluate(query, docs, answer)  # src/retrieval_evaluator.py
    if confidence.overall > threshold:
        break                        # 置信度达标，停止
    elif attempt < max_retries:
        query = rewrite_query(...)   # src/query_rewriter.py，改写查询
```

## 评估器

[src/retrieval_evaluator.py](../src/retrieval_evaluator.py)：`evaluate(query, docs, answer) -> ConfidenceScore`，从 retrieval_confidence / faithfulness / completeness 三维加权计算 overall；失败时降级为基于重排分数的启发式评分，不抛异常阻塞主流程。

## 改写器

[src/query_rewriter.py](../src/query_rewriter.py)：`rewrite_query(original_query, current_query, critique, prev_docs_summary) -> RewrittenQuery`，支持 expand / refine / decompose / rephrase 策略；若改写结果与当前查询相似度>0.9，强制追加限定词避免死循环。

## 输出增强

最终 answer_dict 额外包含 `confidence`（ConfidenceScore）和 `retry_metadata`（每轮 query / confidence / critique / strategy / rationale + forced_exit）字段，经 [app/services.py](../app/services.py) 推送到 SSE 与 [app/schemas.py](../app/schemas.py) 的 `ChatResponse`，并持久化到 [app/db.py](../app/db.py) 的 `chat_history` 表。

## SSE 事件

重试循环发生多轮时推送 `retry_start`（轮数概览）、`retry_reasoning`（每轮查询/置信度/反思/策略）、`retry_success`（最终置信度 + forced_exit）；总超时保持 120s，评估+改写单次控制在 15s 内。事件结构详见 [docs/api.md](api.md#4-sse-事件流)。

## 配置

（[config.json](../config.json) `retry_loop` 段）：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `enable` | `false` | 是否启用重试循环（默认关闭，通过 RunConfig / config.json / KB_RETRY_LOOP__ENABLE 开启） |
| `max_retries` | `2` | 最大重试次数（不含首轮） |
| `threshold` | `0.8` | 综合置信度阈值 |
| `eval_model` | `qwen3.8-max` | 评估器模型 |
| `rewrite_model` | `qwen3.8-max` | 改写器模型 |

> 约束：不修改 src/retrieval.py / src/reranking.py / src/api_requests.py 的现有接口；评估器与改写器均复用 api_requests.py 的 AGICTO 调用封装。
