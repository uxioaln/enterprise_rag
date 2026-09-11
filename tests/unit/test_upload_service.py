# 单元测试：PipelineService 上传入库业务逻辑
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.services import PipelineService


class TestIngestPdfSync:
    """测试 ingest_pdf_sync 的业务逻辑：Mock 掉 MinerU 解析、分块、向量化"""

    @pytest.fixture
    def service(self, mock_pipeline, tmp_data_dir):
        """基于 mock_pipeline 构造 PipelineService"""
        return PipelineService(mock_pipeline)

    def test_ingest_calls_in_correct_order(self, service, mock_pipeline, tmp_data_dir):
        """ingest_pdf_sync 应按正确顺序调用各步骤"""
        # 准备：创建假的分块结果文件（_vectorize_single_report 会检查）
        stem = "新报告"
        chunk_dir = tmp_data_dir / "databases" / "chunked_reports"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_file = chunk_dir / f"{stem}.json"
        chunk_file.write_text(json.dumps(
            {"metainfo": {"sha1": "stock_new"}, "content": {"chunks": [{"text": "测试", "page": 1}]}},
            ensure_ascii=False,
        ))

        file_name = "新报告.pdf"
        company_name = "中芯国际"

        # Mock _vectorize_single_report 避免真实 FAISS 调用
        with patch.object(PipelineService, "_vectorize_single_report") as mock_vec:
            result = service.ingest_pdf_sync(file_name, company_name)

        # 断言返回值结构
        assert result["file_name"] == "新报告.pdf"
        assert result["company_name"] == "中芯国际"
        assert result["sha1"].startswith("stock_")

        # 断言调用顺序：export -> chunk -> vectorize
        mock_pipeline.export_reports_to_markdown.assert_called_once_with(file_name)
        mock_pipeline.chunk_reports.assert_called_once()
        mock_vec.assert_called_once_with(file_name)

    def test_ensure_subset_entry_appends_new(self, service, tmp_data_dir):
        """_ensure_subset_entry 应在 subset.csv 追加新条目"""
        file_name = "全新报告.pdf"
        sha1 = service._ensure_subset_entry(file_name, "中芯国际")

        assert sha1.startswith("stock_")
        # 验证 subset.csv 已追加
        df = pd.read_csv(tmp_data_dir / "subset.csv", encoding="utf-8")
        assert len(df) == 2  # 初始1条 + 新增1条
        assert "全新报告.pdf" in df["file_name"].values
        assert df[df["file_name"] == "全新报告.pdf"].iloc[0]["company_name"] == "中芯国际"

    def test_ensure_subset_entry_reuses_existing_sha1(self, service, tmp_data_dir):
        """_ensure_subset_entry 对已存在文件应复用原 sha1"""
        file_name = "测试报告.pdf"  # 初始 subset.csv 已有
        sha1 = service._ensure_subset_entry(file_name, "中芯国际")

        assert sha1 == "stock_test001"
        # subset.csv 不应新增条目
        df = pd.read_csv(tmp_data_dir / "subset.csv", encoding="utf-8")
        assert len(df) == 1

    def test_ingest_propagates_export_error(self, service, mock_pipeline, tmp_data_dir):
        """export_reports_to_markdown 抛错时应向上传播"""
        mock_pipeline.export_reports_to_markdown.side_effect = RuntimeError("MinerU 解析失败")

        # 准备假的分块文件
        chunk_dir = tmp_data_dir / "databases" / "chunked_reports"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        (chunk_dir / "错误报告.json").write_text('{"metainfo": {"sha1": "x"}, "content": {"chunks": []}}')

        with patch.object(PipelineService, "_vectorize_single_report"):
            with pytest.raises(RuntimeError, match="MinerU 解析失败"):
                service.ingest_pdf_sync("错误报告.pdf", "中芯国际")


class TestVectorizeSingleReport:
    """测试 _vectorize_single_report 的文件检查逻辑"""

    def test_raises_when_chunk_file_missing(self, mock_pipeline, tmp_data_dir):
        """分块结果文件不存在时应抛 FileNotFoundError"""
        service = PipelineService(mock_pipeline)
        with pytest.raises(FileNotFoundError, match="未找到分块结果"):
            service._vectorize_single_report("不存在的文件.pdf")

    def test_calls_ingestor_when_chunk_exists(self, mock_pipeline, tmp_data_dir):
        """分块结果文件存在时应调用 VectorDBIngestor"""
        # 准备假分块文件
        chunk_dir = tmp_data_dir / "databases" / "chunked_reports"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_file = chunk_dir / "存在报告.json"
        chunk_file.write_text(json.dumps(
            {"metainfo": {"sha1": "stock_ok"}, "content": {"chunks": [{"text": "内容"}]}},
            ensure_ascii=False,
        ))

        service = PipelineService(mock_pipeline)

        with patch("app.services.VectorDBIngestor") as MockIngestor:
            mock_ingestor = MockIngestor.return_value
            service._vectorize_single_report("存在报告.pdf")

            MockIngestor.assert_called_once()
            mock_ingestor.process_reports.assert_called_once()
