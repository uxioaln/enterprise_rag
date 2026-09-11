# app/config.py
"""动态配置中心：支持 .env + JSON 配置文件 + 环境变量覆盖，运行时热重载。

设计原则：
1. 配置分层：defaults < config.json < 环境变量（KB_ 前缀）
2. 热重载：通过文件 mtime 检测实现热重载，无需重启服务
3. 类型安全：使用 dataclass(frozen=True, slots=True)，确保不可变与线程安全
4. 关键配置项：Embedding 模型、LLM 端点、模型名称、检索参数
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """模型端点配置（不可变，确保线程安全）。"""
    embedding_model: str = "text-embedding-v4"
    embedding_dim: int = 1024
    embedding_api_base: str = "https://api.agicto.cn/v1"
    embedding_api_key: str = ""
    llm_model: str = "qwen3.8-max"
    llm_api_base: str = "https://api.agicto.cn/v1"
    llm_api_key: str = ""
    llm_fallback_base: str = ""
    llm_fallback_key: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """检索策略配置。"""
    top_n_retrieval: int = 10
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    parent_document_retrieval: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """流水线行为配置。"""
    use_serialized_tables: bool = False
    parallel_requests: int = 1
    answering_model: str = "qwen3.8-max"
    config_suffix: str = ""


@dataclass(frozen=True, slots=True)
class RedisConfig:
    """Redis 连接与 embedding 缓存配置。

    embedding_ttl：embedding 缓存过期时间（秒），通过 SETEX 写入；
    embedding_key_prefix：缓存键前缀，格式为 {prefix}{md5(query_text)}。
    若未来更换 embedding 模型（维度变化），修改该前缀即可避免新旧向量冲突。
    """
    url: str = "redis://localhost:6379/0"
    embedding_ttl: int = 3600
    embedding_key_prefix: str = "emb:v4:"


@dataclass(frozen=True, slots=True)
class RateLimitRule:
    """单业务桶的固定窗口限流规则：window 秒内最多 limit 次。"""
    window: int = 60
    limit: int = 100


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    """AGICTO 出口限流配置（Redis 固定窗口）。

    embedding：embedding 接口限流（默认 60s / 100 次）；
    llm：问答与重排接口限流（默认 60s / 30 次）。
    """
    embedding: RateLimitRule = field(default_factory=lambda: RateLimitRule(limit=100))
    llm: RateLimitRule = field(default_factory=lambda: RateLimitRule(limit=30))


@dataclass(frozen=True, slots=True)
class RetryLoopConfig:
    """低置信度自动查询改写重试配置。

    enable：是否启用重试循环；
    max_retries：最大重试次数（不含首轮，默认 2 即最多 3 轮）；
    threshold：综合置信度阈值，低于此值触发改写重试（0-1）；
    eval_model：评估器使用的模型名；
    rewrite_model：改写器使用的模型名。
    """
    enable: bool = False
    max_retries: int = 2
    threshold: float = 0.8
    eval_model: str = "qwen3.8-max"
    rewrite_model: str = "qwen3.8-max"


@dataclass(frozen=True, slots=True)
class AppConfig:
    """全局应用配置聚合。"""
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    retry_loop: RetryLoopConfig = field(default_factory=RetryLoopConfig)
    loaded_at: float = field(default_factory=time.time)
    source: str = "default"


# --------------------------------------------------------------------------- #
# 配置加载器
# --------------------------------------------------------------------------- #
# 配置文件路径：默认项目根目录的 config.json
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"
# 环境变量前缀，如 KB_LLM_MODEL=gpt-4；嵌套用双下划线，如 KB_MODEL__LLM_MODEL
ENV_PREFIX = "KB_"


def _deep_update(base: dict, override: dict) -> dict:
    """递归合并字典，override 优先级更高。"""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_env_overrides() -> dict:
    """从环境变量加载 KB_ 前缀的配置，支持 __ 表示嵌套。

    自动做布尔/整型/浮点型推断，便于直接落入强类型 dataclass。
    """
    overrides: dict[str, Any] = {}
    for key, val in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        # 去掉前缀并转小写
        path = key[len(ENV_PREFIX):].lower()
        keys = path.split("__")
        # 类型推断：布尔 -> 整型 -> 浮点 -> 字符串
        if val.lower() in ("true", "1", "yes"):
            typed_val: Any = True
        elif val.lower() in ("false", "0", "no"):
            typed_val = False
        else:
            try:
                typed_val = int(val)
            except ValueError:
                try:
                    typed_val = float(val)
                except ValueError:
                    typed_val = val
        # 按 __ 嵌套写入
        target = overrides
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = typed_val
    return overrides


def _build_config_from_dict(data: dict) -> AppConfig:
    """从扁平/嵌套字典构建强类型 AppConfig，忽略未知键以增强兼容性。"""
    model_cfg = ModelConfig(
        **{k: v for k, v in data.get("model", {}).items() if k in ModelConfig.__dataclass_fields__}
    )
    retrieval_cfg = RetrievalConfig(
        **{k: v for k, v in data.get("retrieval", {}).items() if k in RetrievalConfig.__dataclass_fields__}
    )
    pipeline_cfg = PipelineConfig(
        **{k: v for k, v in data.get("pipeline", {}).items() if k in PipelineConfig.__dataclass_fields__}
    )
    redis_cfg = RedisConfig(
        **{k: v for k, v in data.get("redis", {}).items() if k in RedisConfig.__dataclass_fields__}
    )
    # 限流规则按业务桶（embedding / llm）分别解析
    rate_limit_raw = data.get("rate_limit", {})
    embedding_rule = RateLimitRule(
        **{k: v for k, v in rate_limit_raw.get("embedding", {}).items() if k in RateLimitRule.__dataclass_fields__}
    )
    llm_rule = RateLimitRule(
        **{k: v for k, v in rate_limit_raw.get("llm", {}).items() if k in RateLimitRule.__dataclass_fields__}
    )
    rate_limit_cfg = RateLimitConfig(embedding=embedding_rule, llm=llm_rule)
    retry_loop_cfg = RetryLoopConfig(
        **{k: v for k, v in data.get("retry_loop", {}).items() if k in RetryLoopConfig.__dataclass_fields__}
    )
    return AppConfig(
        model=model_cfg,
        retrieval=retrieval_cfg,
        pipeline=pipeline_cfg,
        redis=redis_cfg,
        rate_limit=rate_limit_cfg,
        retry_loop=retry_loop_cfg,
        loaded_at=time.time(),
        source=data.get("_source", "dict"),
    )


async def load_config(
    config_path: Optional[Path] = None,
    force: bool = False,
) -> AppConfig:
    """异步加载配置：defaults <- config.json <- KB_ 环境变量。

    参数 force 当前保留用于显式语义，实际加载始终读取最新文件内容。
    """
    global _current_config, _last_mtime
    config_path = config_path or DEFAULT_CONFIG_PATH

    # 1. 以 dataclass 默认值为基底
    raw: dict[str, Any] = {
        "model": asdict(ModelConfig()),
        "retrieval": asdict(RetrievalConfig()),
        "pipeline": asdict(PipelineConfig()),
        "redis": asdict(RedisConfig()),
        "rate_limit": asdict(RateLimitConfig()),
        "retry_loop": asdict(RetryLoopConfig()),
    }

    # 2. 加载 JSON 配置文件（如果存在）
    source = "default"
    mtime = 0.0
    if config_path.exists():
        async with aiofiles.open(config_path, "r", encoding="utf-8") as f:
            file_content = await f.read()
        file_cfg = json.loads(file_content)
        raw = _deep_update(raw, file_cfg)
        source = str(config_path)
        mtime = config_path.stat().st_mtime

    # 3. 加载 KB_ 前缀环境变量覆盖
    env_cfg = _load_env_overrides()
    if env_cfg:
        raw = _deep_update(raw, env_cfg)
        source += " + env"

    cfg = _build_config_from_dict(raw)
    # frozen dataclass 不可直接赋值，用 object.__setattr__ 写入运行时元信息
    object.__setattr__(cfg, "loaded_at", time.time())
    object.__setattr__(cfg, "source", source)

    _current_config = cfg
    _last_mtime = mtime
    return cfg


def get_config() -> AppConfig:
    """获取当前配置（非异步，用于同步上下文）。"""
    return _current_config


# 模块级单例：首次导入时以全部默认值初始化，应用启动时由 load_config 覆盖
_current_config: AppConfig = _build_config_from_dict({})
_last_mtime: float = 0.0


async def maybe_reload_config(config_path: Optional[Path] = None) -> bool:
    """检查配置文件是否变更，如有则热重载。返回是否发生重载。"""
    config_path = config_path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return False
    current_mtime = config_path.stat().st_mtime
    if current_mtime > _last_mtime:
        await load_config(config_path, force=True)
        return True
    return False


# --------------------------------------------------------------------------- #
# 便捷属性访问（向后兼容，便于同步上下文按字段读取）
# --------------------------------------------------------------------------- #
def embedding_model() -> str:
    return get_config().model.embedding_model


def llm_model() -> str:
    return get_config().model.llm_model


def llm_api_base() -> str:
    return get_config().model.llm_api_base
