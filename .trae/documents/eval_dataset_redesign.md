# RAGAS 评估数据集改进方案

## Context

当前评估数据集 `data/eval/eval_dataset.json` 有 3 个硬伤：
1. **ground truth 来自系统自生成答案**：context_recall 恒为 1.00（系统检索出的上下文自然覆盖自己生成的答案），answer_relevancy 偏低（0.59-0.64，因 ground truth 是 500-1000 字长 CoT 输出）
2. **只有 5 题，单一公司单一类型**：统计不显著，faithfulness/recall 触顶无区分度
3. **1340 题库有 256 个行业宏观问题不含公司名**，走公司路由必定失败

目标：重建评估数据集为 25 题、人工撰写 ground truth、覆盖多公司多题型，让 RAGAS 4 指标恢复区分度。

## 改动清单

### 1. 新建 `data/eval/eval_questions_manual.json`（核心交付物）

25 题人工标注评估数据集，字段：`{id, question, ground_truth, kind, category}`

选题分布：

| 类别 | 题数 | 覆盖能力 | 说明 |
|---|---|---|---|
| A. 年报事实（中芯国际） | 5 | 数值提取 + 跨页聚合 | 营收/毛利率/净利润/EPS/总资产 |
| B. 单券商研报路由 | 7 | 公司路由 badcase | 每家券商 1 题 |
| C. 多券商比较 | 4 | process_comparative_question | 问题含 2 个券商名触发比较 |
| D. 数值提取 | 4 | 数值型区分度 | 目标价/营收预测/产能利用率 |
| E. 跨页聚合 | 3 | context_recall 区分度 | 分地区/分应用收入 |
| F. 行业宏观（带公司路由） | 2 | 宏观信息检索 | 加公司名前缀路由到年报 |

**硬约束**：每题 question 文本必须包含至少 1 个 subset 公司名。

**ground truth 规范**：
- 来源：`data/stock_data/databases/chunked_reports/*.json` 的 chunk text
- 长度：50-300 字，2-3 句话
- 格式：纯文本，无 markdown，无推理链
- 内容：只含核心事实点（数值、名称、关系）

### 2. 修改 `scripts/eval/build_eval_dataset.py`（最小改动）

- 新增常量 `MANUAL_QUESTIONS_PATH = EVAL_DIR / "eval_questions_manual.json"`
- 新增函数 `_load_manual_dataset()`：若人工文件存在则直接读取返回
- 修改 `build_dataset()` 开头：优先调用 `_load_manual_dataset()`，返回 None 时走原有 answers 抽取逻辑
- 原有 `_resolve_answers_file()`、`_extract_records()` 等函数完全保留不动

### 3. 不改动的文件

- `data/stock_data/questions.json` — 评估集与挑战题库是独立数据流
- `scripts/eval/batch_generate.py` — 已兼容 `{id, question, ground_truth, kind}` schema
- `scripts/eval/run_ragas.py` — ground_truth 作为 `SingleTurnSample.reference` 无需改
- `scripts/eval/run_all.py` — 编排顺序不变

## questions.json 是否需要改

**不需要。** 理由：
1. 评估流水线只读 `eval_dataset.json`，不读 `questions.json`
2. 256 个宏观题的路由失败是主流水线问题，与评估数据集改进是独立任务
3. 评估集测"行业宏观"只需在人工文件里给宏观题加公司名前缀

## 实施步骤

1. 读 8 份 chunked_reports JSON，提炼 25 题的事实点
2. 写 `data/eval/eval_questions_manual.json`
3. 预检：每题 question 含至少 1 个 subset 公司名（必须 0 路由失败）
4. 改 `scripts/eval/build_eval_dataset.py`（加常量 + 函数 + 2 行分支）
5. 跑 `build_eval_dataset.py`，确认输出 25 条
6. 单配置 `batch_generate.py --configs base` 验证无 ValueError
7. 跑 `run_ragas.py --configs base`，确认 context_recall 不再恒 1.00
8. 全配置 `run_all.py --configs base,pdr,max`

## 验证标准

- context_recall 不再恒 1.00（应有 < 1 的题）
- answer_relevancy > 0.70（精简 reference 提升）
- faithfulness 仍 > 0.85
- 25 题全部无 ValueError（公司路由预检通过）

## 关键文件

- `data/eval/eval_questions_manual.json`（新建）
- `scripts/eval/build_eval_dataset.py`（修改）
- `data/stock_data/databases/chunked_reports/*.json`（ground truth 数据源，8 份）
- `src/questions_processing.py:273-294`（`_extract_companies_from_subset`，校验用）
