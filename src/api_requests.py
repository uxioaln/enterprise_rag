import os
import json
from dotenv import load_dotenv
from typing import Union, List, Dict, Type, Optional, Literal
from openai import AsyncOpenAI, OpenAI
import asyncio
from src.api_request_parallel_processor import process_api_requests_from_file
from openai.lib._parsing import type_to_response_format_param 
import tiktoken
import src.prompts as prompts
import requests
from json_repair import repair_json
from pydantic import BaseModel
import google.generativeai as genai
from copy import deepcopy
from tenacity import retry, stop_after_attempt, wait_fixed

# OpenAI基础处理器，封装了消息发送、结构化输出、计费等逻辑
class BaseOpenaiProcessor:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.default_model = 'gpt-4o-2024-08-06'
        # self.default_model = 'gpt-4o-mini-2024-07-18',

    def set_up_llm(self):
        # 加载OpenAI API密钥，初始化LLM
        load_dotenv()
        llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=None,
            max_retries=2
            )
        return llm

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None, # For deterministic ouptputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None
        ):
        # 发送消息到OpenAI，支持结构化/非结构化输出
        if model is None:
            model = self.default_model
        params = {
            "model": model,
            "seed": seed,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": human_content}
            ]
        }
        
        # 部分模型不支持temperature
        if "o3-mini" not in model:
            params["temperature"] = temperature
            
        if not is_structured:
            completion = self.llm.chat.completions.create(**params)
            content = completion.choices[0].message.content

        elif is_structured:
            params["response_format"] = response_format
            completion = self.llm.beta.chat.completions.parse(**params)

            response = completion.choices[0].message.parsed
            content = response.dict()

        self.response_data = {"model": completion.model, "input_tokens": completion.usage.prompt_tokens, "output_tokens": completion.usage.completion_tokens}
        print(self.response_data)

        return content

    @staticmethod
    def count_tokens(string, encoding_name="o200k_base"):
        # 统计字符串的token数
        encoding = tiktoken.get_encoding(encoding_name)
        # Encode the string and count the tokens
        tokens = encoding.encode(string)
        token_count = len(tokens)
        return token_count


# IBM API基础处理器，支持余额查询、模型列表、嵌入、消息发送等
class BaseIBMAPIProcessor:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("IBM_API_KEY")
        self.base_url = "https://rag.timetoact.at/ibm"
        self.default_model = 'meta-llama/llama-3-3-70b-instruct'
    def check_balance(self):
        """查询当前API余额"""
        balance_url = f"{self.base_url}/balance"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            response = requests.get(balance_url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error checking balance: {err}")
            return None
    
    def get_available_models(self):
        """获取可用基础模型列表"""
        models_url = f"{self.base_url}/foundation_model_specs"
        
        try:
            response = requests.get(models_url)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting available models: {err}")
            return None
    
    def get_embeddings(self, texts, model_id="ibm/granite-embedding-278m-multilingual"):
        """获取文本的向量嵌入"""
        embeddings_url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": texts,
            "model_id": model_id
        }
        
        try:
            response = requests.post(embeddings_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as err:
            print(f"Error getting embeddings: {err}")
            return None
    
    def send_message(
        self,
        # model='meta-llama/llama-3-1-8b-instruct',
        model=None,
        temperature=0.5,
        seed=None,  # For deterministic outputs
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        max_new_tokens=5000,
        min_new_tokens=1,
        **kwargs
    ):
        # 发送消息到IBM API，支持结构化/非结构化输出
        if model is None:
            model = self.default_model
        text_generation_url = f"{self.base_url}/text_generation"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Prepare the input messages
        input_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": human_content}
        ]
        
        # Prepare parameters with defaults and any additional parameters
        parameters = {
            "temperature": temperature,
            "random_seed": seed,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            **kwargs
        }
        
        payload = {
            "input": input_messages,
            "model_id": model,
            "parameters": parameters
        }
        
        try:
            response = requests.post(text_generation_url, headers=headers, json=payload)
            response.raise_for_status()
            completion = response.json()

            content = completion.get("results")[0].get("generated_text")
            self.response_data = {"model": completion.get("model_id"), "input_tokens": completion.get("results")[0].get("input_token_count"), "output_tokens": completion.get("results")[0].get("generated_token_count")}
            print(self.response_data)
            if is_structured and response_format is not None:
                try:
                    repaired_json = repair_json(content)
                    parsed_dict = json.loads(repaired_json)
                    validated_data = response_format.model_validate(parsed_dict)
                    content = validated_data.model_dump()
                    return content
                
                except Exception as err:
                    print("Error processing structured response, attempting to reparse the response...")
                    reparsed = self._reparse_response(content, system_content)
                    try:
                        repaired_json = repair_json(reparsed)
                        reparsed_dict = json.loads(repaired_json)
                        try:
                            validated_data = response_format.model_validate(reparsed_dict)
                            print("Reparsing successful!")
                            content = validated_data.model_dump()
                            return content
                        
                        except Exception:
                            return reparsed_dict
                        
                    except Exception as reparse_err:
                        print(f"Reparse failed with error: {reparse_err}")
                        print(f"Reparsed response: {reparsed}")
                        return content
            
            return content

        except requests.HTTPError as err:
            print(f"Error generating text: {err}")
            return None

    def _reparse_response(self, response, system_content):

        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=system_content,
            response=response
        )
        
        reparsed_response = self.send_message(
            system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
            human_content=user_prompt,
            is_structured=False
        )
        
        return reparsed_response

     
class BaseGeminiProcessor:
    def __init__(self):
        self.llm = self._set_up_llm()
        self.default_model = 'gemini-2.0-flash-001'
        # self.default_model = "gemini-2.0-flash-thinking-exp-01-21",
        
    def _set_up_llm(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai

    def list_available_models(self) -> None:
        """
        Prints available Gemini models that support text generation.
        """
        print("Available models for text generation:")
        for model in self.llm.list_models():
            if "generateContent" in model.supported_generation_methods:
                print(f"- {model.name}")
                print(f"  Input token limit: {model.input_token_limit}")
                print(f"  Output token limit: {model.output_token_limit}")
                print()

    def _log_retry_attempt(retry_state):
        """Print information about the retry attempt"""
        exception = retry_state.outcome.exception()
        print(f"\nAPI Error encountered: {str(exception)}")
        print("Waiting 20 seconds before retry...\n")

    @retry(
        wait=wait_fixed(20),
        stop=stop_after_attempt(3),
        before_sleep=_log_retry_attempt,
    )
    def _generate_with_retry(self, model, human_content, generation_config):
        """Wrapper for generate_content with retry logic"""
        try:
            return model.generate_content(
                human_content,
                generation_config=generation_config
            )
        except Exception as e:
            if getattr(e, '_attempt_number', 0) == 3:
                print(f"\nRetry failed. Error: {str(e)}\n")
            raise

    def _parse_structured_response(self, response_text, response_format):
        try:
            repaired_json = repair_json(response_text)
            parsed_dict = json.loads(repaired_json)
            validated_data = response_format.model_validate(parsed_dict)
            return validated_data.model_dump()
        except Exception as err:
            print(f"Error parsing structured response: {err}")
            print("Attempting to reparse the response...")
            reparsed = self._reparse_response(response_text, response_format)
            return reparsed

    def _reparse_response(self, response, response_format):
        """Reparse invalid JSON responses using the model itself."""
        user_prompt = prompts.AnswerSchemaFixPrompt.user_prompt.format(
            system_prompt=prompts.AnswerSchemaFixPrompt.system_prompt,
            response=response
        )
        
        try:
            reparsed_response = self.send_message(
                model="gemini-2.0-flash-001",
                system_content=prompts.AnswerSchemaFixPrompt.system_prompt,
                human_content=user_prompt,
                is_structured=False
            )
            
            try:
                repaired_json = repair_json(reparsed_response)
                reparsed_dict = json.loads(repaired_json)
                try:
                    validated_data = response_format.model_validate(reparsed_dict)
                    print("Reparsing successful!")
                    return validated_data.model_dump()
                except Exception:
                    return reparsed_dict
            except Exception as reparse_err:
                print(f"Reparse failed with error: {reparse_err}")
                print(f"Reparsed response: {reparsed_response}")
                return response
        except Exception as e:
            print(f"Reparse attempt failed: {e}")
            return response

    def send_message(
        self,
        model=None,
        temperature: float = 0.5,
        seed=12345,  # For back compatibility
        system_content: str = "You are a helpful assistant.",
        human_content: str = "Hello!",
        is_structured: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, Dict, None]:
        if model is None:
            model = self.default_model

        generation_config = {"temperature": temperature}
        
        prompt = f"{system_content}\n\n---\n\n{human_content}"

        model_instance = self.llm.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        try:
            response = self._generate_with_retry(model_instance, prompt, generation_config)

            self.response_data = {
                "model": response.model_version,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
            print(self.response_data)
            
            if is_structured and response_format is not None:
                return self._parse_structured_response(response.text, response_format)
            
            return response.text
        except Exception as e:
            raise Exception(f"API request failed after retries: {str(e)}")


class APIProcessor:
    def __init__(self, provider: Literal["openai", "ibm", "gemini", "dashscope"] ="dashscope"):
        self.provider = provider.lower()
        if self.provider == "openai":
            self.processor = BaseOpenaiProcessor()
        elif self.provider == "ibm":
            self.processor = BaseIBMAPIProcessor()
        elif self.provider == "gemini":
            self.processor = BaseGeminiProcessor()
        elif self.provider == "dashscope":
            self.processor = BaseDashscopeProcessor()

    def send_message(
        self,
        model=None,
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        human_content="Hello!",
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        Routes the send_message call to the appropriate processor.
        The underlying processor's send_message method is responsible for handling the parameters.
        """
        if model is None:
            model = self.processor.default_model
        return self.processor.send_message(
            model=model,
            temperature=temperature,
            seed=seed,
            system_content=system_content,
            human_content=human_content,
            is_structured=is_structured,
            response_format=response_format,
            **kwargs
        )

    def get_answer_from_rag_context(self, question, rag_context, schema, model):
        system_prompt, response_format, user_prompt = self._build_rag_context_prompts(schema)
        
        answer_dict = self.processor.send_message(
            model=model,
            system_content=system_prompt,
            human_content=user_prompt.format(context=rag_context, question=question),
            is_structured=True,
            response_format=response_format
        )
        self.response_data = self.processor.response_data
        
        # 检查返回的字典是否包含所需的字段，如果不是dashscope则进行兜底
        if not isinstance(answer_dict, dict) or 'step_by_step_analysis' not in answer_dict:
            # 如果是dashscope返回的基本格式，尝试保留其内容
            if isinstance(answer_dict, dict) and 'final_answer' in answer_dict:
                # 这是dashscope处理后的格式，尝试从final_answer中提取结构化信息
                final_answer_content = answer_dict.get("final_answer", "N/A")
                
                # 如果final_answer是字符串且包含结构化信息，尝试解析
                if isinstance(final_answer_content, str) and final_answer_content.strip().startswith('{'):
                    try:
                        structured_data = json.loads(final_answer_content)
                        answer_dict = structured_data
                    except json.JSONDecodeError:
                        # 如果final_answer不是JSON，保持原有结构
                        answer_dict = {
                            "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                            "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                            "relevant_pages": answer_dict.get("relevant_pages", []),
                            "final_answer": answer_dict.get("final_answer", "N/A")
                        }
                else:
                    # 否则使用兜底结构
                    answer_dict = {
                        "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
                        "reasoning_summary": answer_dict.get("reasoning_summary", ""),
                        "relevant_pages": answer_dict.get("relevant_pages", []),
                        "final_answer": answer_dict.get("final_answer", "N/A")
                    }
            else:
                # 如果不是预期格式，进行兜底
                answer_dict = {
                    "step_by_step_analysis": "",
                    "reasoning_summary": "",
                    "relevant_pages": [],
                    "final_answer": "N/A"
                }
        return answer_dict


    def _build_rag_context_prompts(self, schema):
        """Return prompts tuple for the given schema."""
        use_schema_prompt = True if self.provider == "ibm" or self.provider == "gemini" else False
        
        if schema == "name":
            system_prompt = (prompts.AnswerWithRAGContextNamePrompt.system_prompt_with_schema 
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamePrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamePrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamePrompt.user_prompt
        elif schema == "number":
            system_prompt = (prompts.AnswerWithRAGContextNumberPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNumberPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNumberPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNumberPrompt.user_prompt
        elif schema == "boolean":
            system_prompt = (prompts.AnswerWithRAGContextBooleanPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextBooleanPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextBooleanPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextBooleanPrompt.user_prompt
        elif schema == "names":
            system_prompt = (prompts.AnswerWithRAGContextNamesPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextNamesPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextNamesPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextNamesPrompt.user_prompt
        elif schema == "comparative":
            system_prompt = (prompts.ComparativeAnswerPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.ComparativeAnswerPrompt.system_prompt)
            response_format = prompts.ComparativeAnswerPrompt.AnswerSchema
            user_prompt = prompts.ComparativeAnswerPrompt.user_prompt
        elif schema == "string":
            # 新增：支持开放性文本问题
            system_prompt = (prompts.AnswerWithRAGContextStringPrompt.system_prompt_with_schema
                            if use_schema_prompt else prompts.AnswerWithRAGContextStringPrompt.system_prompt)
            response_format = prompts.AnswerWithRAGContextStringPrompt.AnswerSchema
            user_prompt = prompts.AnswerWithRAGContextStringPrompt.user_prompt
        else:
            raise ValueError(f"Unsupported schema: {schema}")
        return system_prompt, response_format, user_prompt

    def get_rephrased_questions(self, original_question: str, companies: List[str]) -> Dict[str, str]:
        """Use LLM to break down a comparative question into individual questions."""
        answer_dict = self.processor.send_message(
            system_content=prompts.RephrasedQuestionsPrompt.system_prompt,
            human_content=prompts.RephrasedQuestionsPrompt.user_prompt.format(
                question=original_question,
                companies=", ".join([f'"{company}"' for company in companies])
            ),
            is_structured=True,
            response_format=prompts.RephrasedQuestionsPrompt.RephrasedQuestions
        )
        
        # Convert the answer_dict to the desired format
        questions_dict = {item["company_name"]: item["question"] for item in answer_dict["questions"]}
        
        return questions_dict


class AsyncOpenaiProcessor:
    
    def _get_unique_filepath(self, base_filepath):
        """Helper method to get unique filepath"""
        if not os.path.exists(base_filepath):
            return base_filepath
        
        base, ext = os.path.splitext(base_filepath)
        counter = 1
        while os.path.exists(f"{base}_{counter}{ext}"):
            counter += 1
        return f"{base}_{counter}{ext}"

    async def process_structured_ouputs_requests(
        self,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.5,
        seed=None,
        system_content="You are a helpful assistant.",
        queries=None,
        response_format=None,
        requests_filepath='./temp_async_llm_requests.jsonl',
        save_filepath='./temp_async_llm_results.jsonl',
        preserve_requests=False,
        preserve_results=True,
        request_url="https://api.openai.com/v1/chat/completions",
        max_requests_per_minute=3_500,
        max_tokens_per_minute=3_500_000,
        token_encoding_name="o200k_base",
        max_attempts=5,
        logging_level=20,
        progress_callback=None
    ):
        # Create requests for jsonl
        jsonl_requests = []
        for idx, query in enumerate(queries):
            request = {
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query},
                ],
                'response_format': type_to_response_format_param(response_format),
                'metadata': {'original_index': idx}
            }
            jsonl_requests.append(request)
            
        # Get unique filepaths if files already exist
        requests_filepath = self._get_unique_filepath(requests_filepath)
        save_filepath = self._get_unique_filepath(save_filepath)

        # Write requests to JSONL file
        with open(requests_filepath, "w") as f:
            for request in jsonl_requests:
                json_string = json.dumps(request)
                f.write(json_string + "\n")

        # Process API requests
        total_requests = len(jsonl_requests)

        async def monitor_progress():
            last_count = 0
            while True:
                try:
                    with open(save_filepath, 'r') as f:
                        current_count = sum(1 for _ in f)
                        if current_count > last_count:
                            if progress_callback:
                                for _ in range(current_count - last_count):
                                    progress_callback()
                            last_count = current_count
                        if current_count >= total_requests:
                            break
                except FileNotFoundError:
                    pass
                await asyncio.sleep(0.1)

        async def process_with_progress():
            await asyncio.gather(
                process_api_requests_from_file(
                    requests_filepath=requests_filepath,
                    save_filepath=save_filepath,
                    request_url=request_url,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_requests_per_minute=max_requests_per_minute,
                    max_tokens_per_minute=max_tokens_per_minute,
                    token_encoding_name=token_encoding_name,
                    max_attempts=max_attempts,
                    logging_level=logging_level
                ),
                monitor_progress()
            )

        await process_with_progress()

        with open(save_filepath, "r") as f:
            validated_data_list = []
            results = []
            for line_number, line in enumerate(f, start=1):
                raw_line = line.strip()
                try:
                    result = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_number}: Failed to load JSON from line: {raw_line}")
                    continue

                # Check finish_reason in the API response
                finish_reason = result[1]['choices'][0].get('finish_reason', '')
                if finish_reason != "stop":
                    print(f"[WARNING] Line {line_number}: finish_reason is '{finish_reason}' (expected 'stop').")

                # Safely parse answer; if it fails, leave answer empty and report the error.
                try:
                    answer_content = result[1]['choices'][0]['message']['content']
                    answer_parsed = json.loads(answer_content)
                    answer = response_format(**answer_parsed).model_dump()
                except Exception as e:
                    print(f"[ERROR] Line {line_number}: Failed to parse answer JSON. Error: {e}.")
                    answer = ""

                results.append({
                    'index': result[2],
                    'question': result[0]['messages'],
                    'answer': answer
                })
            
            # Sort by original index and build final list
            validated_data_list = [
                {'question': r['question'], 'answer': r['answer']} 
                for r in sorted(results, key=lambda x: x['index']['original_index'])
            ]

        if not preserve_requests:
            os.remove(requests_filepath)

        if not preserve_results:
            os.remove(save_filepath)
        else:  # Fix requests order
            with open(save_filepath, "r") as f:
                results = [json.loads(line) for line in f]
            
            sorted_results = sorted(results, key=lambda x: x[2]['original_index'])
            
            with open(save_filepath, "w") as f:
                for result in sorted_results:
                    json_string = json.dumps(result)
                    f.write(json_string + "\n")
            
        return validated_data_list

# AGICTO（OpenAI 兼容接口）基础处理器，通过 AGICTO 站点调用大模型
class BaseDashscopeProcessor:
    def __init__(self):
        # 从环境变量读取 AGICTO API-KEY 与 BASE_URL
        load_dotenv()
        self.api_key = os.getenv("AGICTO_API_KEY")
        self.base_url = "https://api.agicto.cn/v1"
        self.default_model = 'qwen3.8-max'
        self.llm = self.set_up_llm()

    def set_up_llm(self):
        # 初始化 AGICTO 兼容的 OpenAI 客户端
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=None,
            max_retries=2
        )

    def send_message(
        self,
        model="qwen3.8-max",
        temperature=0.1,
        seed=None,  # 兼容参数，暂不使用
        system_content='You are a helpful assistant.',
        human_content='Hello!',
        is_structured=False,
        response_format=None,
        **kwargs
    ):
        """
        发送消息到 AGICTO（OpenAI 兼容接口）大模型，支持 system_content + human_content 拼接为 messages。
        暂不支持结构化输出。
        """
        if model is None:
            model = self.default_model
        # 出口限流：问答/重排等所有 AGICTO LLM 调用共用 llm 桶（Redis 异常自动放行）
        from app.rate_limiter import check_rate_limit_sync
        check_rate_limit_sync("llm")
        # 拼接 messages
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if human_content:
            messages.append({"role": "user", "content": human_content})

        # 调用 AGICTO（OpenAI 兼容接口）chat.completions.create
        completion = self.llm.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        print('AGICTO api_key=', self.api_key)
        print('AGICTO base_url=', self.base_url)
        print('model=', model)
        print('response=', completion)
        # 兼容 dashscope/openai/gemini 返回格式，始终返回 dict
        content = completion.choices[0].message.content
        # 增加 response_data 属性，保证接口一致性
        self.response_data = {
            "model": completion.model,
            "input_tokens": completion.usage.prompt_tokens if completion.usage else None,
            "output_tokens": completion.usage.completion_tokens if completion.usage else None
        }
        print('content=', content)

        # 尝试解析 content 为 JSON，如果是结构化响应
        try:
            # 先尝试移除可能的markdown代码块标记
            content_str = content.strip()
            if content_str.startswith('```') and '```' in content_str[3:]:
                # 找到第一个 ``` 和 最后一个 ``` 之间的内容
                first_backtick = content_str.find('```') + 3
                next_newline = content_str.find('\n', first_backtick)
                if next_newline > 0:
                    first_backtick = next_newline + 1
                last_backtick = content_str.rfind('```')
                if last_backtick > first_backtick:
                    json_str = content_str[first_backtick:last_backtick].strip()
                else:
                    json_str = content_str
            else:
                json_str = content_str

            # 尝试解析 JSON
            parsed_content = json.loads(json_str)
            return parsed_content
        except (json.JSONDecodeError, TypeError):
            # 如果不是有效的JSON，返回基本格式
            print(f"Content is not valid JSON, returning basic format: {content}")
            return {"final_answer": content, "step_by_step_analysis": "", "reasoning_summary": "", "relevant_pages": []}


# --------------------------------------------------------------------------- #
# AGICTO 统一出口：embedding（Redis 缓存 + 固定窗口限流）与 chat（限流）
#
# 缓存策略：键 {prefix}{md5(text)}（prefix 来自 config.json redis.embedding_key_prefix，
# 默认 emb:v4:），SETEX 写入 TTL 默认 3600 秒；命中直接返回，不消耗限流配额。
# 降级策略：Redis 未启用或连接异常时自动直连原链路，不阻断服务。
# --------------------------------------------------------------------------- #
# AGICTO embedding 单次请求批量上限
# 注意：text-embedding-v4 模型服务端限制单批不超过 10 条文本，
# 超过会返回 HTTP 200 但 data 为 null（batch size is invalid），故此处必须为 10
_AGICTO_EMBED_BATCH_SIZE = 10
# AGICTO 默认模型
AGICTO_DEFAULT_CHAT_MODEL = "qwen3.8-max"
AGICTO_DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"

# 模块级懒加载客户端（复用 HTTP 连接池）
_agicto_client: Optional[OpenAI] = None
_agicto_async_client: Optional[AsyncOpenAI] = None


def _get_agicto_client() -> OpenAI:
    """懒加载 AGICTO 同步客户端（OpenAI 兼容接口）。"""
    global _agicto_client
    if _agicto_client is None:
        load_dotenv()
        _agicto_client = OpenAI(
            api_key=os.getenv("AGICTO_API_KEY"),
            base_url="https://api.agicto.cn/v1",
            timeout=None,
            max_retries=2
        )
    return _agicto_client


def _get_agicto_async_client() -> AsyncOpenAI:
    """懒加载 AGICTO 异步客户端（OpenAI 兼容接口）。"""
    global _agicto_async_client
    if _agicto_async_client is None:
        load_dotenv()
        _agicto_async_client = AsyncOpenAI(
            api_key=os.getenv("AGICTO_API_KEY"),
            base_url="https://api.agicto.cn/v1",
            timeout=None,
            max_retries=2
        )
    return _agicto_async_client


def _call_agicto_embeddings(texts: List[str], model: str = AGICTO_DEFAULT_EMBEDDING_MODEL) -> List[List[float]]:
    """直接调用 AGICTO embedding 接口（OpenAI 兼容），内部按批切分。

    返回与 texts 等长的向量列表（响应 data 顺序与输入顺序一致）。
    """
    client = _get_agicto_client()
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), _AGICTO_EMBED_BATCH_SIZE):
        batch = texts[i:i + _AGICTO_EMBED_BATCH_SIZE]
        try:
            resp = client.embeddings.create(model=model, input=batch)
        except TypeError as exc:
            # AGICTO 对非法请求（如批量超过 10 条、参数错误）会返回 HTTP 200 但 data 为 null，
            # openai 库解析时迭代 None 抛出含糊的 TypeError，这里转换为可读错误
            raise RuntimeError(
                f"AGICTO embedding 响应异常（本批 {len(batch)} 条）：可能批量超过上限({_AGICTO_EMBED_BATCH_SIZE})或参数非法。"
                f"原始错误: {exc}"
            ) from exc
        if not getattr(resp, "data", None):
            raise RuntimeError(f"AGICTO embedding 未返回数据（本批 {len(batch)} 条），请检查批量大小与模型参数")
        for emb_data in resp.data:
            embedding = emb_data.embedding
            if embedding is None or len(embedding) == 0:
                raise RuntimeError("AGICTO返回的embedding为空")
            embeddings.append(embedding)
    return embeddings


def get_embeddings(texts: Union[str, List[str]], model: str = AGICTO_DEFAULT_EMBEDDING_MODEL) -> List[List[float]]:
    """AGICTO embedding 统一出口：缓存命中直接返回，未命中时限流后调用并回写缓存。

    流程（Redis 启用时）：
      1. 批量 GET 缓存（命中不消耗限流配额）
      2. 对未命中的文本执行固定窗口限流检查（rl:agicto:embed）
      3. 调用 AGICTO text-embedding-v4（同步 HTTP 放入线程，避免阻塞桥接事件循环）
      4. SETEX 回写缓存（TTL 来自配置 redis.embedding_ttl）

    Redis 未启用或连接失败时自动降级为直接调用，不阻断服务。
    始终返回与输入等长的向量列表；输入为单个字符串时返回长度为 1 的列表。
    """
    from app import cache as kb_cache

    single_input = isinstance(texts, str)
    text_list = [texts] if single_input else list(texts)
    if not text_list or not all(isinstance(t, str) and t.strip() for t in text_list):
        raise ValueError("待嵌入文本必须为非空字符串")

    if not kb_cache.is_enabled():
        # Redis 未启用（CLI 场景或初始化失败）：缓存与限流降级，直接走原链路
        return _call_agicto_embeddings(text_list, model)

    from app import rate_limiter as kb_rate_limiter

    async def _flow(client):
        # 1. 批量读缓存
        hits = await kb_cache.get_cached_embeddings(text_list, redis=client)
        results: List[Optional[List[float]]] = list(hits)
        miss_idx = [i for i, vec in enumerate(results) if vec is None]
        if miss_idx:
            # 2. 限流检查：仅真实调用前消耗配额（桶名需与 rate_limiter 一致："embedding"）
            await kb_rate_limiter.check_rate_limit("embedding", redis=client)
            # 3. 调用 AGICTO（同步 HTTP 放线程池）
            miss_texts = [text_list[i] for i in miss_idx]
            miss_vectors = await asyncio.to_thread(_call_agicto_embeddings, miss_texts, model)
            for i, vec in zip(miss_idx, miss_vectors):
                results[i] = vec
            # 4. SETEX 回写缓存
            await kb_cache.set_cached_embeddings(
                [(text_list[i], results[i]) for i in miss_idx], redis=client
            )
        return results

    flow_result = kb_cache.run_with_redis(_flow)
    if flow_result is kb_cache.DEGRADED:
        # Redis 连接失败：降级为直连
        return _call_agicto_embeddings(text_list, model)
    return flow_result


def chat_completion(
    model: str = AGICTO_DEFAULT_CHAT_MODEL,
    temperature: float = 0.1,
    system_content: str = "You are a helpful assistant.",
    human_content: str = "Hello!",
) -> str:
    """AGICTO chat 同步出口：先执行 llm 桶限流检查，再调用接口，返回回复文本。"""
    from app.rate_limiter import check_rate_limit_sync
    check_rate_limit_sync("llm")
    client = _get_agicto_client()
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    if human_content:
        messages.append({"role": "user", "content": human_content})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return completion.choices[0].message.content


async def async_chat_completion(
    model: str = AGICTO_DEFAULT_CHAT_MODEL,
    temperature: float = 0.1,
    system_content: str = "You are a helpful assistant.",
    human_content: str = "Hello!",
) -> str:
    """AGICTO chat 异步出口：先执行 llm 桶限流检查（全局 Redis 连接），再异步调用接口。"""
    from app.rate_limiter import check_rate_limit
    await check_rate_limit("llm")
    client = _get_agicto_async_client()
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    if human_content:
        messages.append({"role": "user", "content": human_content})
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return completion.choices[0].message.content
