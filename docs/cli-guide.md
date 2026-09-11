# CLI 与离线流水线运行指南

本指南覆盖不使用 Docker / FastAPI 服务时的运行方式：CLI 分步执行（方式 A）、直接运行 pipeline 脚本（方式 B）、单条问题即时推理（方式 C），以及数据准备与输出格式说明。

## 1. 环境准备

```bash
git clone <your-repo-url>
cd 企业知识库_new
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -e . -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 配置 API Key

将 `env` 文件重命名为 `.env` 并填入真实凭证（完整模板见 [README 快速开始](../README.md#快速开始)），要点：

- 所有 LLM 与嵌入调用均通过 AGICTO 平台统一进行，仅需配置 `AGICTO_API_KEY`。
- MinerU 云端解析的 API Key 当前内置在 [src/pdf_mineru.py](../src/pdf_mineru.py)，如需替换为自己的 Key，请修改该文件中的 `api_key` 变量。
- OSS 凭证用于 `/upload` 接口：上传的 PDF 会先传到 OSS 公网路径，MinerU 再通过该 URL 读取并解析。

## 3. 准备数据

将 PDF 研报放入 `data/stock_data/pdf_reports/`，并维护 `data/stock_data/subset.csv`：

```csv
file_name,company_name,sha1
【国信证券】工业与汽车触底反弹，良率影响短期营收.pdf,国信证券,stock_10001
【财报】中芯国际：中芯国际2024年年度报告.pdf,中芯国际,stock_10002
...
```

`sha1` 字段作为 FAISS 索引文件名，建议用稳定唯一标识（如 `stock_1000X`）。

在 `data/stock_data/questions.json` 中维护问题列表：

```json
[
  {
    "text": "\"中芯国际\"在晶圆制造行业中的地位如何？",
    "kind": "string"
  }
]
```

> 问题中公司名需用英文双引号包住，便于公司路由抽取。

## 4. 数据集说明

当前 `data/stock_data/` 下包含 9 份中芯国际相关 PDF：

- 7 份券商深度研究报告（上海证券、东方证券、中原证券、光大证券、兴证国际、华泰证券、国信证券）
- 1 份年度报告（中芯国际 2024 年年度报告）
- 1 份机构调研纪要

示例问题（见 `data/stock_data/questions.json`）：

- "中芯国际"在晶圆制造行业中的地位如何？其服务范围和全球布局是怎样的？
- 半导体行业有哪些关键特性，这些特性如何助力"中芯国际"发展？
- "中芯国际"的营收和利润情况近期有何变化？影响因素是什么？
- "中芯国际"的收入结构有何变化？尤其是在中国大陆和北美市场的表现如何？
- 美国对中国半导体产业的限制政策对"中芯国际"有何影响？"中芯国际"如何应对？

## 5. 运行方式

### 方式 A：CLI（推荐）

通过 [main.py](../main.py) 暴露的 click 命令分步执行：

```bash
# 1) 解析 PDF（MinerU 云端解析，支持并行）
python main.py parse-pdfs --parallel --chunk-size 2 --max-workers 10

# 2) 表格序列化（可选，启用 use_serialized_tables 时需要）
python main.py serialize-tables --max-workers 10

# 3) 规整报告 -> 分块 -> 入库
python main.py process-reports --config no_ser_tab

# 4) 处理问题并生成答案（可选配置：base / pdr / max）
python main.py process-questions --config max
```

### 方式 B：直接运行 pipeline 脚本

编辑 [src/pipeline.py](../src/pipeline.py) 末尾的 `__main__` 块，按需取消注释各步骤：

```bash
python src/pipeline.py
```

完整流程：

1. **PDF 解析**：遍历 `pdf_reports/` 下所有 PDF，调用 MinerU 解析为 Markdown + content_list
2. **分块**：基于 `content_list.json` 的 `page_idx` 真实页码分块
3. **入库**：生成 FAISS 向量索引 + BM25 索引
4. **问答**：处理 `questions.json`，生成 `answers_*.json`

### 方式 C：单条问题即时推理

在 Python 中调用 `Pipeline.answer_single_question` 即可获得单条问题的结构化答案：

```python
from pyprojroot import here
from src.pipeline import Pipeline, max_config

pipeline = Pipeline(here() / "data" / "stock_data", run_config=max_config)
answer = pipeline.answer_single_question("\"中芯国际\"在晶圆制造行业中的地位如何？", kind="string")
print(answer)
```

### 方式 D：FastAPI 服务

生产环境推荐使用 FastAPI 服务方式，参见 [README 快速开始](../README.md#快速开始) 与 [docs/api.md](api.md)。

## 6. 输出格式

`answers_*.json` 中每条答案为结构化 JSON，便于下游展示与评估：

```json
{
  "question": "\"中芯国际\"在晶圆制造行业中的地位如何？",
  "kind": "string",
  "step_by_step_analysis": "...分步推理过程...",
  "reasoning_summary": "...推理摘要...",
  "relevant_pages": [
    {"file_name": "【上海证券】中芯国际深度研究报告....pdf", "page": 3},
    {"file_name": "【华泰证券】中芯国际（688981）....pdf", "page": 5}
  ],
  "final_answer": "...最终答案..."
}
```
