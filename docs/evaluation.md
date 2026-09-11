# RAGAS 评估指南

项目提供一套一键式 RAGAS 评估流水线，自动完成"数据构造 -> 批量推理 -> 指标计算 -> Bad Case 分析"全流程，并输出可视化报告。脚本位于 [scripts/eval/](../scripts/eval/)。

## 评估指标定义

| 指标 | 含义 | 关注点 |
|---|---|---|
| `faithfulness` | 忠实度 | 生成答案是否完全由检索上下文支撑，是否存在幻觉 |
| `answer_relevancy` | 答案相关性 | 生成答案是否紧扣问题意图 |
| `context_precision` | 上下文精度 | 检索结果中相关 chunk 的占比，衡量是否混入噪声（重排质量） |
| `context_recall` | 上下文召回 | ground truth 是否被检索上下文覆盖，衡量是否漏召 |

## 一键运行

前置条件：已为待评估配置构建好向量库（`databases_<suffix>/vector_dbs`），并已在 `.env` 中配置 `AGICTO_API_KEY`。

```bash
# 一键运行全流程（默认评估 base / pdr / max 三种配置）
python scripts/eval/run_all.py

# 仅评估指定配置
python scripts/eval/run_all.py --configs base,max
```

流水线按顺序执行四个阶段：

| 阶段 | 脚本 | 作用 |
|---|---|---|
| 1. 数据构造 | [build_eval_dataset.py](../scripts/eval/build_eval_dataset.py) | 读取 `answers_<max>.json` 提取 question / ground_truth，采样（>10 条随机抽 10 条），输出 `data/eval/eval_dataset.json` |
| 2. 批量推理 | [batch_generate.py](../scripts/eval/batch_generate.py) | 对 base/pdr/max 三种配置逐条调用 `Pipeline.answer_with_contexts`，输出 `data/eval/results_<config>.json`（含 answer / contexts / relevant_pages，单条失败记录 error 不中断） |
| 3. 指标计算 | [run_ragas.py](../scripts/eval/run_ragas.py) | 用 AGICTO（`qwen3.8-max` + `text-embedding-v4`）初始化 RAGAS 评估器，计算四项指标，回写 `results_<config>.json` 的 `scores`，并生成 `scores.json` 与 `scores_table.md` |
| 4. Bad Case | [analyze_badcase.py](../scripts/eval/analyze_badcase.py) | 按阈值筛选失败案例，挑选 3-5 个代表性案例，诊断根因，生成 `bad_cases.md` |

## 评估结果

- 配置对比表：[data/eval/scores_table.md](../data/eval/scores_table.md)（行：base/pdr/max，列：四项指标，保留 2 位小数，附结论）
- 聚合分数：[data/eval/scores.json](../data/eval/scores.json)
- Bad Case 分析：[data/eval/bad_cases.md](../data/eval/bad_cases.md)，包含基本信息、内容对比（Ground Truth / Generated Answer / 上下文摘要）、根因标签（检索漏召 / 重排错位 / 幻觉 / 答案偏题 / 检索失败）与详细分析，以及基于共性的优化建议

Bad Case 判定阈值：`faithfulness < 0.6` 或 `answer_relevancy < 0.6` 或 `context_precision < 0.5` 或 `context_recall < 0.5`。

## 依赖说明

评估依赖 RAGAS（`ragas>=0.2.0,<0.3`）与 `langchain-openai`，需与项目既有 `langchain==0.3.3` / `openai==1.51.2` 版本兼容，详见 [requirements.txt](../requirements.txt)。运行前请确保已为各配置构建向量库，否则检索阶段会失败并记录为 Bad Case。
