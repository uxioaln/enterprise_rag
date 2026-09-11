import os
import json
import pickle
import threading
from typing import List, Union
from pathlib import Path
from tqdm import tqdm
import hashlib

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt, RetryError

# BM25Ingestor：BM25索引构建与保存工具
class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """从文本块列表创建BM25索引"""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """
        批量处理所有报告，生成并保存BM25索引。
        参数：
            all_reports_dir (Path): 存放JSON报告的目录
            output_dir (Path): 保存BM25索引的目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # 加载报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # 提取文本块并创建BM25索引
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # 保存BM25索引，文件名用sha1_name
            sha1_name = report_data["metainfo"]["sha1"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

# VectorDBIngestor：向量库构建与保存工具
class VectorDBIngestor:
    # 类级锁：FAISS 索引更新（add + 落盘）必须串行化，
    # 多线程/Celery 线程池并发调用 add_chunks_to_index_atomic 时互斥
    _index_lock = threading.Lock()

    def __init__(self):
        # 初始化 AGICTO（OpenAI 兼容接口）客户端
        load_dotenv()
        self.embedding_client = OpenAI(
            api_key=os.getenv("AGICTO_API_KEY"),
            base_url="https://api.agicto.cn/v1",
            timeout=None,
            max_retries=2
        )

    def _get_embeddings(self, text: Union[str, List[str]], model: str = "text-embedding-v4") -> List[float]:
        """获取文本或文本块的嵌入向量，支持重试。

        包装层：tenacity 重试耗尽后抛出的 RetryError 内部持有 Future 对象，
        无法被 Celery 序列化（会变成 UnpickleableExceptionWrapper，前端拿不到真实原因）。
        这里统一转换为普通 RuntimeError，保留最后一次异常的可读信息。
        """
        try:
            return self._get_embeddings_with_retry(text, model)
        except RetryError as exc:
            # 取最后一次尝试的真实异常（如 APIConnectionError / 限流等），字符串化后抛出
            last_exception = exc.last_attempt.exception() if exc.last_attempt else exc
            raise RuntimeError(f"调用 embedding 接口失败（已重试）: {last_exception}")
        except Exception as exc:
            raise RuntimeError(f"调用 embedding 接口失败: {exc}")

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings_with_retry(self, text: Union[str, List[str]], model: str = "text-embedding-v4") -> List[float]:
        # 获取文本或文本块的嵌入向量，支持重试
        # 通过 src.api_requests.get_embeddings 统一出口调用 AGICTO text-embedding-v4：
        # 带 Redis 缓存与固定窗口限流（命中不消耗配额），Redis 异常自动降级为直连
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        # 保证 input 为一维字符串列表或单个字符串
        if isinstance(text, list):
            text_chunks = text
        else:
            text_chunks = [text]

        # 类型检查，确保每一项都是字符串
        if not all(isinstance(x, str) for x in text_chunks):
            raise ValueError("所有待嵌入文本必须为字符串类型！实际类型: {}".format([type(x) for x in text_chunks]))

        # 过滤空字符串
        text_chunks = [x for x in text_chunks if x.strip()]
        if not text_chunks:
            raise ValueError("所有待嵌入文本均为空字符串！")
        print('start embedding ================================')

        # 统一出口获取嵌入（内部按 25 条/批切分调用）
        from src.api_requests import get_embeddings
        embeddings = get_embeddings(text_chunks, model=model)

        # 逐条校验嵌入结果，空值写入日志（保留原有校验逻辑）
        LOG_FILE = 'embedding_error.log'
        for idx, embedding in enumerate(embeddings):
            if embedding is None or len(embedding) == 0:
                error_text = text_chunks[idx] if idx < len(text_chunks) else None
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(f"AGICTO返回的embedding为空，index={idx}，文本内容如下：\n{error_text}\n{'-'*60}\n")
                raise RuntimeError(f"AGICTO返回的embedding为空，index={idx}，文本内容已写入 {LOG_FILE}")
        return embeddings

    def add_chunks_to_index_atomic(
        self,
        chunks: List[str],
        vectors: List[List[float]],
        index_path: Union[str, Path, None] = None,
        append: bool = False,
    ) -> "faiss.Index":
        # 线程安全地把向量加入 FAISS 索引并原子落盘。
        # 锁内执行 index.add() 与写盘，确保并发更新时索引不崩溃、文件不损坏。
        # 参数：
        #   chunks: 文本块列表（与 vectors 一一对应，供日志与扩展使用，FAISS 只需向量）
        #   vectors: 嵌入向量列表
        #   index_path: faiss 文件路径；为 None 时仅返回内存索引不落盘
        #   append: True 时加载已有索引并追加；False（默认）时新建索引（覆盖写，与原行为一致）
        if not vectors:
            raise ValueError("vectors 不能为空，无法构建 FAISS 索引")
        vectors_array = np.array(vectors, dtype=np.float32)

        with VectorDBIngestor._index_lock:
            if append and index_path is not None and Path(index_path).exists():
                # 加载已有索引，追加向量
                index = faiss.read_index(str(index_path))
                if index.d != vectors_array.shape[1]:
                    raise ValueError(
                        f"向量维度不匹配：已有索引维度 {index.d}，新向量维度 {vectors_array.shape[1]}"
                    )
            else:
                # 新建索引，采用内积（余弦距离），与原 _create_vector_db 一致
                index = faiss.IndexFlatIP(vectors_array.shape[1])
            index.add(vectors_array)
            if index_path is not None:
                # 先写临时文件再原子替换，避免并发写入产生损坏文件
                index_path = Path(index_path)
                index_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = index_path.with_suffix(".faiss.tmp")
                faiss.write_index(index, str(tmp_path))
                tmp_path.replace(index_path)
            return index

    def _create_vector_db(self, embeddings: List[float]):
        # 用faiss构建向量库，采用内积（余弦距离）
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        # 针对单份报告，提取文本块并生成嵌入向量
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        # 过滤空内容，超长内容截断到 2048 字符
        max_len = 2048
        text_chunks = [t[:max_len] for t in text_chunks if len(t) > 0]
        embeddings = self._get_embeddings(text_chunks)
        return text_chunks, embeddings

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        # 批量处理所有报告，生成并保存faiss向量库
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports for FAISS"):
            # 加载报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            text_chunks, embeddings = self._process_report(report_data)
            # 用 metainfo['sha1'] 作为 faiss 文件名，避免中文和特殊字符
            sha1 = report_data["metainfo"].get("sha1", "")
            if not sha1:
                raise ValueError(f"分块报告 {report_path} 缺少 sha1 字段，无法保存 faiss 文件！")
            faiss_file_path = output_dir / f"{sha1}.faiss"
            # 加锁构建索引并原子落盘（默认新建覆盖，与原行为一致）
            self.add_chunks_to_index_atomic(text_chunks, embeddings, index_path=faiss_file_path)

        print(f"Processed {len(all_report_paths)} reports")