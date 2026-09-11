# RAGAS 评估分数汇总

| 配置 | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| base | 0.98 | 0.59 | 0.79 | 1.00 |
| pdr | 1.00 | 0.64 | 0.82 | 1.00 |
| max | 1.00 | 0.59 | 0.82 | 1.00 |

### 结论

- faithfulness：`pdr` 配置表现最佳（1.00）
- answer_relevancy：`pdr` 配置表现最佳（0.64）
- context_precision：`max` 配置表现最佳（0.82）
- context_recall：`base` 配置表现最佳（1.00）
