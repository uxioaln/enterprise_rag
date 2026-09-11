# AGICTO 平台（OpenAI 兼容接口）调用大模型
from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd
import shutil
import threading
import time

# pdf为中文，适用mineru,docling适用英文；download_docling_models/parse_pdf_reports_parallel 依赖 PDFParser
# PDFParser 依赖可选库 docling：缺失时导入为 None，仅离线 PDF 解析脚本受影响，运行时问答服务不依赖它
try:
    from src.pdf_parsing import PDFParser
except ImportError:
    logging.getLogger(__name__).warning(
        "未检测到 docling 依赖，PDFParser 不可用；离线 PDF 解析脚本将无法使用，运行时问答服务不受影响"
    )
    PDFParser = None
from src import pdf_mineru
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
from src.tables_serialization import TableSerializer

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        # 路径配置，支持不同流程和数据目录
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name
        
        self.answers_file_path = root_path / f"answers{config_suffix}.json"       
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"
        
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        # self.parsed_reports_dirname = "01_parsed_reports"
        # self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        # self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"
        # MinerU content_list 抽取目录：保留每份报告的 *_content_list.json 和 images/
        self.reports_content_list_dirname = f"03_reports_content_list{suffix}"

        #self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        #self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        #self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname
        self.reports_content_list_path = self.debug_data_path / self.reports_content_list_dirname

@dataclass
class RunConfig:
    # 运行流程参数配置
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 1 # 并行的数量，需要限制，否则qwen3.8-max会超出阈值
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = "dashscope" #openai
    answering_model: str = "qwen3.8-max" # gpt-4o-mini-2024-07-18 or "gpt-4o-2024-08-06"
    config_suffix: str = ""
    # 低置信度自动查询改写重试配置
    enable_retry_loop: bool = False
    max_retries: int = 2
    confidence_threshold: float = 0.8
    retry_eval_model: str = "qwen3.8-max"
    retry_rewrite_model: str = "qwen3.8-max"

class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        # 初始化主流程，加载路径和配置
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()
        # 线程本地缓存：单次推理中 _retrieve_contexts 与 _generate_answer 间共享检索状态，
        # 保证并发调用（FastAPI 线程池）互不干扰，同时保留页码校验行为
        self._tl = threading.local()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """根据配置初始化所有路径"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在subset.json且无subset.csv，若是则自动转换为CSV。
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"
        
        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                df = pd.DataFrame(data)
                
                df.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    @staticmethod
    def download_docling_models():
        # 下载Docling所需模型，避免首次运行时自动下载
        if PDFParser is None:
            raise RuntimeError(
                "download_docling_models 依赖 docling，当前环境未安装。"
                "请先安装 docling（pip install docling）后再执行该命令。"
            )
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """多进程并行解析PDF报告，提升处理效率
        参数：
            chunk_size: 每个worker处理的PDF数
            num_workers: 并发worker数
        """
        if PDFParser is None:
            raise RuntimeError(
                "parse_pdf_reports_parallel 依赖 docling 进行 PDF 解析，当前环境未安装 docling。"
                "请先安装 docling（pip install docling）后重试。"
            )
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))
        
        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def export_reports_to_markdown(self, file_name):
        """
        使用 pdf_mineru.py，将指定 PDF 文件转换为 markdown，并放到 reports_markdown_dirname 目录下。
        同步将 content_list.json 与 images/ 抽取到 reports_content_list_dirname 目录下，供后续按真实页码分块使用。
        :param file_name: PDF 文件名（如 '【财报】中芯国际：中芯国际2024年年度报告.pdf'）
        """
        # 方式二（签名 URL 直传文件）：申请 MinerU 上传链接并直传本地 PDF（无需 OSS），
        # 返回 batch_id，再轮询解析结果并下载、解压（返回解压目录）
        local_pdf_path = self.paths.pdf_reports_dir / file_name
        if not local_pdf_path.is_file():
            print(f"本地 PDF 文件不存在: {local_pdf_path}")
            return
        print(f"开始处理: {file_name}")
        batch_id = pdf_mineru.get_batch_id(str(local_pdf_path))
        print(f"batch_id: {batch_id}")
        extract_dir = pdf_mineru.get_result(batch_id)
        if not extract_dir or not os.path.isdir(extract_dir):
            print(f"未找到解压目录: {extract_dir}")
            return

        # 在解压目录中定位 full.md
        md_path = os.path.join(extract_dir, "full.md")
        if not os.path.exists(md_path):
            # 兜底：可能解压目录有嵌套，glob 查找
            import glob
            candidates = glob.glob(os.path.join(extract_dir, "**", "full.md"), recursive=True)
            if candidates:
                md_path = candidates[0]
            else:
                print(f"未找到 markdown 文件: {md_path}")
                return

        # 目标 markdown 目录
        os.makedirs(self.paths.reports_markdown_path, exist_ok=True)
        # 目标文件名为原始 file_name，扩展名改为 .md
        base_name = os.path.splitext(file_name)[0]
        target_path = os.path.join(self.paths.reports_markdown_path, f"{base_name}.md")
        shutil.move(md_path, target_path)
        print(f"已将 {md_path} 移动到 {target_path}")

        # 同步抽取 content_list.json 与 images/ 到 reports_content_list_path
        # 用 base_name 作为前缀命名 content_list 文件（与 subset.csv 的 file_name 去扩展名一致，
        # 这样 split_reports_from_content_list 才能查到对应的 sha1 / company_name）
        os.makedirs(self.paths.reports_content_list_path, exist_ok=True)
        pdf_mineru.export_content_list(
            extract_dir=extract_dir,
            target_dir=self.paths.reports_content_list_path,
            file_name=base_name
        )

    def chunk_reports(self, include_serialized_tables: bool = False):
        """
        将规整后报告分块，便于后续向量化和检索。
        优先使用 MinerU content_list.json 切分（页码来自 page_idx，真实准确，包含 text/table/image）。
        若 reports_content_list_path 为空，则 fallback 到 split_markdown_reports（按行估算页码）。
        """
        text_splitter = TextSplitter()
        content_list_dir = self.paths.reports_content_list_path
        use_content_list = (
            content_list_dir is not None
            and content_list_dir.exists()
            and any(content_list_dir.glob("*_content_list.json"))
        )

        if use_content_list:
            print(f"基于 MinerU content_list.json 分块（真实页码）: {content_list_dir}")
            text_splitter.split_reports_from_content_list(
                content_list_dir=content_list_dir,
                output_dir=self.paths.documents_dir,
                subset_csv=self.paths.subset_path
            )
            # 逐报告补齐：部分报告可能只有 markdown 而无 content_list（如 MinerU 仅导出 md、
            # content_list 缺失的年报）。对这些报告回退按 markdown 行数估算页码分块，
            # 已有 content_list 分块的报告通过 skip_file_names 跳过，避免覆盖真实页码。
            markdown_dir = self.paths.reports_markdown_path
            if markdown_dir is not None and markdown_dir.exists():
                content_list_stems = {
                    p.name.replace("_content_list.json", "")
                    for p in content_list_dir.glob("*_content_list.json")
                    if not p.name.endswith("_content_list_v2.json")
                }
                markdown_stems = {p.stem for p in markdown_dir.glob("*.md")}
                missing_stems = markdown_stems - content_list_stems
                if missing_stems:
                    print(f"以下报告无 content_list，回退到按 markdown 行估算分块: {sorted(missing_stems)}")
                    text_splitter.split_markdown_reports(
                        all_md_dir=markdown_dir,
                        output_dir=self.paths.documents_dir,
                        subset_csv=self.paths.subset_path,
                        skip_file_names=content_list_stems
                    )
        else:
            # Fallback：按 markdown 行数估算页码
            print(f"未发现 content_list.json，回退到按 markdown 行估算分块: {self.paths.reports_markdown_path}")
            text_splitter.split_markdown_reports(
                all_md_dir=self.paths.reports_markdown_path,
                output_dir=self.paths.documents_dir,
                subset_csv=self.paths.subset_path
            )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """从分块报告创建向量数据库"""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir
        
        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")
    
    def create_bm25_db(self):
        """从分块报告创建BM25数据库"""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path
        
        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")
    
    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        # 解析PDF报告，支持并行处理
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)

    def process_parsed_reports(self):
        """
        处理已解析的PDF报告，主要流程：
        1. 对报告进行分块
        2. 创建向量数据库
        """
        print("开始处理报告流程...")
        
        print("步骤1：报告分块...")
        self.chunk_reports()
        
        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()
        
        print("报告处理流程已成功完成！")
        
    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        获取下一个可用的文件名，如果文件已存在则自动添加编号后缀。
        例如：若answers.json已存在，则返回answers_01.json等。
        """
        if not base_path.exists():
            return base_path
            
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename
            
            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        # 处理所有问题，生成答案文件
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context            
        )
        
        output_path = self._get_next_available_filename(self.paths.answers_file_path)
        
        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    def _new_single_question_processor(self) -> QuestionsProcessor:
        """构造用于单条问题推理的 QuestionsProcessor（parallel_requests=1，避免 AGICTO 限流）"""
        return QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=None,  # 单问无需文件
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=1,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )

    def _retrieve_contexts(self, question: str) -> tuple[list[str], list[int]]:
        """检索上下文，返回 (纯文本片段列表, 页码列表)。
        检索结果与路由信息缓存到线程本地存储，供 _generate_answer 复用，
        以保留与原 answer_single_question 一致的页码校验与引用提取行为。
        """
        t0 = time.time()
        print("[计时] 开始初始化 QuestionsProcessor ...")
        processor = self._new_single_question_processor()
        t1 = time.time()
        print(f"[计时] QuestionsProcessor 初始化耗时: {t1-t0:.2f} 秒")
        print("[计时] 开始检索上下文 ...")
        retrieval_results, company_name, is_comparative, companies = processor.retrieve_question_contexts(question)
        t2 = time.time()
        print(f"[计时] 检索上下文耗时: {t2-t1:.2f} 秒")
        # 缓存到线程本地存储，供 _generate_answer 复用
        self._tl.retrieval_results = retrieval_results
        self._tl.company_name = company_name
        self._tl.is_comparative = is_comparative
        self._tl.companies = companies
        contexts = [r["text"] for r in retrieval_results]
        pages = [r.get("page", 0) for r in retrieval_results]
        return contexts, pages

    def _generate_answer(self, question: str, contexts: list[str], kind: str = "string") -> dict:
        """根据上下文生成答案，返回结构化 answer_dict。
        复用本线程 _retrieve_contexts 缓存的检索结果与公司名，确保页码校验与引用提取
        行为与原 answer_single_question 一致；多公司比较则委托原比较流程。
        """
        t0 = time.time()
        processor = self._new_single_question_processor()
        retrieval_results = getattr(self._tl, "retrieval_results", None)
        company_name = getattr(self._tl, "company_name", None)
        is_comparative = getattr(self._tl, "is_comparative", False)
        companies = getattr(self._tl, "companies", None)
        print("[计时] 开始生成答案 ...")
        if is_comparative and companies:
            # 多公司比较：复用原比较流程，保留引用聚合等原行为
            answer = processor.process_comparative_question(question, companies, kind)
        else:
            answer = processor.generate_answer_from_contexts(
                question,
                contexts,
                kind,
                company_name=company_name,
                retrieval_results=retrieval_results
            )
        t1 = time.time()
        print(f"[计时] 生成答案耗时: {t1-t0:.2f} 秒")
        # 清理线程本地缓存，避免跨调用残留
        for _k in ("retrieval_results", "company_name", "is_comparative", "companies"):
            setattr(self._tl, _k, None)
        return answer

    def answer_with_contexts(self, question: str, kind: str = "string") -> tuple[dict, list[str]]:
        """单条问题推理，返回 (answer_dict, contexts_text_list)。
        contexts_text_list 为纯文本列表，顺序与 Prompt 中一致，供 RAGAS 评估使用。
        内部调用 _retrieve_contexts 与 _generate_answer 两个方法。
        若 RunConfig.enable_retry_loop 为 True，则包进低置信度自动查询改写重试循环。
        """
        # 未启用重试循环：保持原有单趟逻辑
        if not self.run_config.enable_retry_loop:
            contexts, _pages = self._retrieve_contexts(question)
            answer_dict = self._generate_answer(question, contexts, kind)
            return answer_dict, contexts

        # 启用重试循环：retrieve -> generate -> evaluate -> 判断阈值 -> rewrite
        return self._answer_with_retry_loop(question, kind)

    def _answer_with_retry_loop(self, question: str, kind: str = "string") -> tuple[dict, list[str]]:
        """低置信度自动查询改写重试循环。

        循环内顺序：retrieve -> generate -> evaluate -> 判断阈值 ->（未达标则）rewrite_query。
        最终输出 answer_dict 增加 confidence 和 retry_metadata 字段。
        """
        from src.retrieval_evaluator import evaluate, ConfidenceScore
        from src.query_rewriter import rewrite_query, RewrittenQuery

        original_query = question
        current_query = question
        max_retries = self.run_config.max_retries
        threshold = self.run_config.confidence_threshold
        eval_model = self.run_config.retry_eval_model
        rewrite_model = self.run_config.retry_rewrite_model

        # 每轮的元数据记录
        retry_metadata: list[dict] = []
        best_answer_dict: dict = {}
        best_contexts: list[str] = []
        best_confidence: ConfidenceScore | None = None
        forced_exit = False

        for attempt in range(max_retries + 1):
            t_round = time.time()
            print(f"[重试循环] 第 {attempt+1}/{max_retries+1} 轮，查询: {current_query}")

            # 1. retrieve
            contexts, _pages = self._retrieve_contexts(current_query)

            # 2. generate
            answer_dict = self._generate_answer(current_query, contexts, kind)

            # 3. evaluate
            confidence = evaluate(current_query, contexts, answer_dict, model=eval_model)
            print(f"[重试循环] 第 {attempt+1} 轮置信度: overall={confidence.overall:.2f}, "
                  f"retrieval={confidence.retrieval_confidence:.2f}, "
                  f"faithfulness={confidence.faithfulness:.2f}, "
                  f"completeness={confidence.completeness:.2f}")

            # 记录本轮元数据
            round_meta = {
                "attempt": attempt + 1,
                "query": current_query,
                "confidence": confidence.to_dict(),
                "critique": confidence.critique,
                "strategy": "",
                "rationale": "",
            }
            retry_metadata.append(round_meta)

            # 保留最佳结果（置信度最高的一轮）
            if best_confidence is None or confidence.overall > best_confidence.overall:
                best_answer_dict = dict(answer_dict)
                best_contexts = list(contexts)
                best_confidence = confidence

            # 4. 判断阈值
            if confidence.overall > threshold:
                print(f"[重试循环] 置信度 {confidence.overall:.2f} > 阈值 {threshold}，停止重试")
                forced_exit = False
                break
            elif attempt < max_retries:
                # 5. rewrite_query
                docs_summary = "\n".join(contexts[:3])[:1000] if contexts else ""
                rewritten = rewrite_query(
                    original_query=original_query,
                    current_query=current_query,
                    critique=confidence.critique,
                    prev_docs_summary=docs_summary,
                    model=rewrite_model,
                )
                print(f"[重试循环] 改写结果: {rewritten.rewritten_query} (策略: {rewritten.rewrite_strategy})")
                # 更新下一轮的查询与元数据
                current_query = rewritten.rewritten_query
                retry_metadata[-1]["strategy"] = rewritten.rewrite_strategy
                retry_metadata[-1]["rationale"] = rewritten.rationale
                round_elapsed = time.time() - t_round
                print(f"[重试循环] 第 {attempt+1} 轮耗时: {round_elapsed:.2f} 秒")
            else:
                print(f"[重试循环] 已达最大重试次数 {max_retries}，使用最佳结果")
                forced_exit = True

        # 最终输出增加 confidence 和 retry_metadata 字段
        best_answer_dict["confidence"] = best_confidence.to_dict() if best_confidence else {}
        best_answer_dict["retry_metadata"] = {
            "rounds": retry_metadata,
            "forced_exit": forced_exit,
            "total_attempts": len(retry_metadata),
            "final_confidence": best_confidence.to_dict() if best_confidence else {},
        }
        return best_answer_dict, best_contexts

    def answer_single_question(self, question: str, kind: str = "string"):
        """
        单条问题即时推理，返回结构化答案（dict）。
        重构后内部调用 _retrieve_contexts 与 _generate_answer，保持原有签名与返回值不变。
        若 RunConfig.enable_retry_loop 为 True，answer_with_contexts 内部自动启用重试循环，
        返回的 answer_dict 将额外包含 confidence 和 retry_metadata 字段。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        t0 = time.time()
        answer_dict, _contexts = self.answer_with_contexts(question, kind=kind)
        t1 = time.time()
        print(f"[计时] answer_single_question 总耗时: {t1-t0:.2f} 秒")
        return answer_dict

preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),
                      "no_ser_tab": RunConfig(use_serialized_tables=False)}

base_config = RunConfig(
    parallel_requests=10,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
    config_suffix="_base"
)

parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=20,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_pdr"
)

## 这里
max_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=4,
    submission_file=True,
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = qwen3.8-max",
    answering_model="qwen3.8-max",
    config_suffix="_kimi_k2_5"
)


configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config}


# 你可以直接在本文件中运行任意方法：
# python .\src\pipeline.py
# 只需取消你想运行的方法的注释即可
# 你也可以修改 run_config 以尝试不同的配置
if __name__ == "__main__":
    # 设置数据集根目录（此处以 test_set 为例）
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)
    #print(type(root_path))
    # 初始化主流程，使用推荐的最佳配置
    pipeline = Pipeline(root_path, run_config=max_config)
    
    print('1. 将pdf转化为纯markdown文本')#使用mineru的云端解析结果下载得到压缩包，解压后得到的文件为md格式
    # 遍历 pdf_reports 目录下所有 PDF 文件，依次调用 export_reports_to_markdown
    pdf_reports_dir = pipeline.paths.pdf_reports_dir
    pdf_files = sorted(pdf_reports_dir.glob("*.pdf"))
    print(f"待处理 PDF 数量: {len(pdf_files)}")
    for pdf_path in pdf_files:
        print(f"--- 处理: {pdf_path.name} ---")
        pipeline.export_reports_to_markdown(pdf_path.name)

    # 5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports
    print('2. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    pipeline.chunk_reports() 
    
    # 6. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    print('3. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
    pipeline.create_vector_dbs()     
    
    # 7. 处理问题并生成答案，具体逻辑取决于 run_config
    # 默认questions.json
    print('4. 处理问题并生成答案，具体逻辑取决于 run_config')
    pipeline.process_questions() 
    
    print('完成')
