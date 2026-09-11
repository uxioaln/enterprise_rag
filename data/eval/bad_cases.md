# Bad Case 分析报告

共筛选 bad case 15 个，下文展示 5 个代表性案例。

判定阈值：faithfulness < 0.6 或 answer_relevancy < 0.6 或 context_precision < 0.5 或 context_recall < 0.5。

### 案例 1 [base] — 根因：检索失败

- **问题**："中芯国际"在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？
- **分数**：faithfulness=N/A, answer_relevancy=N/A, context_precision=N/A, context_recall=N/A
- **相关页面**：[]

**内容对比**：

- **Ground Truth（截断 200 字）**：N/A
- **Generated Answer（截断 200 字）**：
- **上下文摘要（前 3 个 chunk）**：

- （无上下文）

**诊断**：检索失败

推理阶段抛出异常：ValueError: No report found with '中芯国际' company name.，未能生成答案或上下文。

### 案例 1 [pdr] — 根因：检索失败

- **问题**："中芯国际"在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？
- **分数**：faithfulness=N/A, answer_relevancy=N/A, context_precision=N/A, context_recall=N/A
- **相关页面**：[]

**内容对比**：

- **Ground Truth（截断 200 字）**：N/A
- **Generated Answer（截断 200 字）**：
- **上下文摘要（前 3 个 chunk）**：

- （无上下文）

**诊断**：检索失败

推理阶段抛出异常：ValueError: No report found with '中芯国际' company name.，未能生成答案或上下文。

### 案例 1 [max] — 根因：检索失败

- **问题**："中芯国际"在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？
- **分数**：faithfulness=N/A, answer_relevancy=N/A, context_precision=N/A, context_recall=N/A
- **相关页面**：[]

**内容对比**：

- **Ground Truth（截断 200 字）**：N/A
- **Generated Answer（截断 200 字）**：
- **上下文摘要（前 3 个 chunk）**：

- （无上下文）

**诊断**：检索失败

推理阶段抛出异常：ValueError: No report found with '中芯国际' company name.，未能生成答案或上下文。

### 案例 2 [base] — 根因：检索失败

- **问题**：半导体行业有哪些关键特性，这些特性如何助力"中芯国际"发展？
- **分数**：faithfulness=N/A, answer_relevancy=N/A, context_precision=N/A, context_recall=N/A
- **相关页面**：[]

**内容对比**：

- **Ground Truth（截断 200 字）**：N/A
- **Generated Answer（截断 200 字）**：
- **上下文摘要（前 3 个 chunk）**：

- （无上下文）

**诊断**：检索失败

推理阶段抛出异常：ValueError: No report found with '中芯国际' company name.，未能生成答案或上下文。

### 案例 3 [base] — 根因：检索失败

- **问题**："中芯国际"的营收和利润情况近期有何变化？影响因素是什么？
- **分数**：faithfulness=N/A, answer_relevancy=N/A, context_precision=N/A, context_recall=N/A
- **相关页面**：[]

**内容对比**：

- **Ground Truth（截断 200 字）**：N/A
- **Generated Answer（截断 200 字）**：
- **上下文摘要（前 3 个 chunk）**：

- （无上下文）

**诊断**：检索失败

推理阶段抛出异常：ValueError: No report found with '中芯国际' company name.，未能生成答案或上下文。

## 优化建议（基于共性根因）

- **检索失败**（5 例）：检查对应配置的向量库是否已构建（databases_<suffix>/vector_dbs），确认 subset.csv 中公司名与文档 metainfo 一致。
