import os
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI
import requests
from json_repair import repair_json
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor

_log = logging.getLogger(__name__)


# JinaReranker：基于Jina API的重排器，适用于多语言场景
class JinaReranker:
    def __init__(self):
        # 初始化Jina重排API地址和请求头
        self.url = 'https://api.jina.ai/v1/rerank'
        self.headers = self.get_headers()
        
    def get_headers(self):
        # 加载Jina API密钥，组装请求头
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {'Content-Type': 'application/json',
                   'Authorization': f'Bearer {jina_api_key}'}
        return headers
    
    def rerank(self, query, documents, top_n = 10):
        # 调用Jina API进行重排，返回top_n相关文档
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()

# LLMReranker：基于大模型的重排器，支持单条和批量重排
class LLMReranker:
    def __init__(self, provider: str = "agicto"):
        # 支持 openai/agicto（AGICTO 使用 OpenAI 兼容接口），默认 agicto
        self.provider = provider.lower()
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
        self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks

    def set_up_llm(self):
        # 根据 provider 初始化 LLM 客户端
        load_dotenv()
        if self.provider == "openai":
            return OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=300,  # 重排生成类调用超时 300 秒，避免服务端挂起时无限等待
                max_retries=2
            )
        elif self.provider == "agicto":
            # 通过 AGICTO 的 OpenAI 兼容接口调用
            return OpenAI(
                api_key=os.getenv("AGICTO_API_KEY"),
                base_url="https://api.agicto.cn/v1",
                timeout=300,  # 重排生成类调用超时 300 秒，避免 AGICTO 挂起时无限等待
                max_retries=2
            )
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    @staticmethod
    def _parse_single_block_json(content: str) -> dict:
        """解析 AGICTO(qwen-plus) 单块重排返回的 JSON 字符串。

        qwen-plus 不支持 structured output，返回的是 JSON 字符串，用 json_repair
        容错解析后提取 relevance_score；解析失败时降级为中性分 0.5，避免 0.0
        导致该文档被强烈压低（退化为纯向量分数排序更稳妥）。
        """
        try:
            parsed = json.loads(repair_json(content))
            if isinstance(parsed, dict) and "relevance_score" in parsed:
                # clamp 到 [0, 1]，防止模型返回越界值
                score = max(0.0, min(1.0, float(parsed["relevance_score"])))
                return {
                    "relevance_score": score,
                    "reasoning": str(parsed.get("reasoning", "")),
                }
        except Exception as e:
            _log.warning(f"AGICTO 单块重排 JSON 解析失败，降级为中性分 0.5: {e}")
        return {"relevance_score": 0.5, "reasoning": "解析失败，降级为中性分"}

    @staticmethod
    def _parse_multiple_blocks_json(content: str, expected_count: int) -> dict:
        """解析 AGICTO(qwen-plus) 多块重排返回的 JSON 字符串。

        返回结构与 openai 分支保持一致：{"block_rankings": [{...}, ...]}；
        解析失败或数量不足时补中性分 0.5，保证长度与输入文本块数量一致。
        """
        default_ranking = {"relevance_score": 0.5, "reasoning": "解析失败，降级为中性分"}
        try:
            parsed = json.loads(repair_json(content))
            if isinstance(parsed, dict) and isinstance(parsed.get("block_rankings"), list):
                result = []
                for item in parsed["block_rankings"][:expected_count]:
                    if isinstance(item, dict):
                        score = max(0.0, min(1.0, float(item.get("relevance_score", 0.5))))
                        result.append({
                            "relevance_score": score,
                            "reasoning": str(item.get("reasoning", "")),
                        })
                # 数量不足时用中性分补齐，保证与输入文本块一一对应
                while len(result) < expected_count:
                    result.append(default_ranking.copy())
                return {"block_rankings": result}
        except Exception as e:
            _log.warning(f"AGICTO 多块重排 JSON 解析失败，降级为中性分 0.5: {e}")
        return {"block_rankings": [default_ranking.copy() for _ in range(expected_count)]}

    def get_rank_for_single_block(self, query, retrieved_document):
        # 针对单个文本块，调用LLM进行相关性评分
        # 修正原 /n 笔误为 \n，使换行符真正生效
        user_prompt = f'\nHere is the query:\n"{query}"\n\nHere is the retrieved text block:\n"""\n{retrieved_document}\n"""\n'
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_single_block},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_single_block
            )
            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            return response_dict
        elif self.provider == "agicto":
            # 通过 AGICTO OpenAI 兼容接口调用 qwen-plus
            # qwen-plus 仅支持 response_format={"type":"json_object"}，不支持 json_schema strict
            messages = [
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ]
            completion = self.llm.chat.completions.create(
                model="qwen-plus",
                temperature=0,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or ""
            # 用 json_repair 解析 qwen-plus 返回的 JSON 字符串，提取 relevance_score
            return self._parse_single_block_json(content)
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        # 针对多个文本块，批量调用LLM进行相关性评分
        formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""\n{text}\n"""' for i, text in enumerate(retrieved_documents)])
        user_prompt = (
            f"Here is the query: \"{query}\"\n\n"
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order."
        )
        if self.provider == "openai":
            completion = self.llm.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                temperature=0,
                messages=[
                    {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=self.schema_for_multiple_blocks
            )
            response = completion.choices[0].message.parsed
            response_dict = response.model_dump()
            return response_dict
        elif self.provider == "agicto":
            # 通过 AGICTO OpenAI 兼容接口调用 qwen-plus
            # qwen-plus 仅支持 response_format={"type":"json_object"}，不支持 json_schema strict
            messages = [
                {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                {"role": "user", "content": user_prompt},
            ]
            completion = self.llm.chat.completions.create(
                model="qwen-plus",
                temperature=0,
                messages=messages,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or ""
            # 用 json_repair 解析 qwen-plus 返回的 JSON 字符串，提取每个块的 relevance_score
            return self._parse_multiple_blocks_json(content, len(retrieved_documents))
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def rerank_documents(self, query: str, documents: list, documents_batch_size: int = 4, llm_weight: float = 0.7):
        """
        使用多线程并行方式对多个文档进行重排。
        结合向量相似度和LLM相关性分数，采用加权平均融合。
        参数：
            query: 查询语句
            documents: 待重排的文档列表，每个元素需包含'text'和'similarity'
                （similarity 统一语义：越大越相关；兼容旧 'distance' 字段的输入）
            documents_batch_size: 每批送入LLM的文档数
            llm_weight: LLM分数权重（0-1），其余为向量分数权重
        返回：
            按融合分数降序排序的文档列表
        """
        # 按batch分组
        doc_batches = [documents[i:i + documents_batch_size] for i in range(0, len(documents), documents_batch_size)]
        vector_weight = 1 - llm_weight

        # 统一相似度语义并归一化（修复 distance 语义不一致问题）：
        # 1. 读取 similarity 字段（统一"越大越相关"语义）；旧格式数据无该字段时回落 distance
        # 2. min-max 归一化到 [0,1]，消除不同检索器的量纲差异
        #    （BM25 分数无界、FAISS IndexFlatIP 内积约 0~1，原始分数直接加权会让量纲大的一方主导融合）
        # 3. 所有候选相似度相同（max==min）时统一取 0.5 中性值，避免除零
        sims = [float(d.get('similarity', d.get('distance', 0.0))) for d in documents]
        sim_min, sim_max = min(sims), max(sims)
        if sim_max > sim_min:
            norm_sims = [(s - sim_min) / (sim_max - sim_min) for s in sims]
        else:
            norm_sims = [0.5] * len(sims)

        if documents_batch_size == 1:
            def process_single_doc(doc, norm_sim):
                # 单文档重排（norm_sim 为该文档归一化后的相似度，越大越相关）
                ranking = self.get_rank_for_single_block(query, doc['text'])

                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # 计算融合分数：similarity 越大越相关，归一化后与 LLM 分数同量纲
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"] +
                    vector_weight * norm_sim,
                    4
                )
                return doc_with_score

            # 多线程并行处理，max_workers=1 保证 dashscope LLM 串行调用，避免 QPS 超限
            with ThreadPoolExecutor(max_workers=1) as executor:
                all_results = list(executor.map(process_single_doc, documents, norm_sims))

        else:
            def process_batch(batch, norm_batch):
                # 批量重排（norm_batch 为该批文档归一化后的相似度列表，与 batch 一一对应）
                texts = [doc['text'] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get('block_rankings', [])

                if len(block_rankings) < len(batch):
                    print(f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}")
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(f"Missing ranking for document on page {doc.get('page', 'unknown')}:")
                        print(f"Text preview: {doc['text'][:100]}...\n")

                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append({
                            "relevance_score": 0.0,
                            "reasoning": "Default ranking due to missing LLM response"
                        })

                for doc, rank, norm_sim in zip(batch, block_rankings, norm_batch):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    # 计算融合分数：similarity 越大越相关，归一化后与 LLM 分数同量纲
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"] +
                        vector_weight * norm_sim,
                        4
                    )
                    results.append(doc_with_score)
                return results

            # 将归一化相似度按同样的 batch 尺寸切片，与 doc_batches 一一对应
            norm_batches = [norm_sims[i:i + documents_batch_size] for i in range(0, len(norm_sims), documents_batch_size)]
            # 多线程并行处理，max_workers=1 保证 dashscope LLM 串行调用，避免 QPS 超限
            with ThreadPoolExecutor(max_workers=1) as executor:
                batch_results = list(executor.map(process_batch, doc_batches, norm_batches))
            
            # 扁平化结果
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
        
        # 按融合分数降序排序
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
