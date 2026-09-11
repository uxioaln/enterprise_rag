# FastAPI 应用入口：lifespan 初始化 Pipeline 与持久化存储，挂载路由与全局中间件
import asyncio
import gc
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pyprojroot import here

from app import cache
from app.api import router
from app.config import load_config, maybe_reload_config
from app.db import init_db
from app.middleware import ErrorHandlingMiddleware, RequestLoggingMiddleware
from app.services import PipelineService
from app.storage import get_storage
from src.pipeline import Pipeline, max_config

logger = logging.getLogger(__name__)

# 开发环境默认允许的前端来源（本地常见 dev server 端口）
_DEV_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
]


def _resolve_cors_origins() -> list[str]:
    """解析允许的跨域来源。

    优先读取环境变量 ALLOWED_ORIGINS（逗号分隔，如 "https://a.com,https://b.com"）；
    未设置时回退到开发环境默认的本地前端地址。生产环境务必显式配置 ALLOWED_ORIGINS。
    """
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return list(_DEV_ORIGINS)


def _add_cors_middleware(application: FastAPI) -> None:
    """挂载全局 CORSMiddleware。

    CORS 规范禁止 allow_credentials=True 与通配符 "*" 同时使用，因此当来源解析为
    ["*"] 时自动关闭 credentials，避免运行时报错；其余情况（具体来源）均开启
    credentials，保证 SSE 端点 /chat 的预检（OPTIONS）能正常返回 200。
    """
    origins = _resolve_cors_origins()
    allow_credentials = origins != ["*"]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化数据库、加载配置、构建 Pipeline 与存储，关闭时释放资源"""
    # 初始化 SQLite 表结构（幂等，可安全重复调用）
    await init_db()
    # 加载配置（config.json + KB_ 环境变量），后续可由 _config_watcher 热重载
    await load_config()
    # 初始化 Redis 连接池（embedding 缓存 + 限流依赖）；
    # 连接失败仅降级为无缓存/无限流模式，不阻断服务启动
    redis_ok = await cache.init_redis()
    logger.info("Redis 初始化%s", "成功" if redis_ok else "失败（缓存与限流已降级）")
    # 初始化 Celery：导入任务模块完成注册，确保 /upload 可提交异步任务、
    # worker 可独立启动（celery -A app.celery_app:celery_app worker）
    from app import tasks as _celery_tasks  # noqa: F401
    from app.celery_app import celery_app as _celery_app  # noqa: F401
    app.state.celery_app = _celery_app
    logger.info("数据库表结构与配置已就绪，Celery 任务已注册")

    # 数据根目录与 CLI 保持一致：data/stock_data
    root_path = here() / "data" / "stock_data"
    logger.info("初始化 Pipeline，数据目录: %s", root_path)
    # 使用推荐的最佳配置（父文档检索 + LLM 重排 + qwen3.8-max）
    pipeline = Pipeline(root_path, run_config=max_config)
    app.state.pipeline_service = PipelineService(pipeline)
    # 通过工厂函数选择存储后端（STORAGE_BACKEND：sqlite / memory）
    app.state.storage = get_storage()
    # 启动后台协程：周期检测配置文件变更并热重载
    app.state.config_reload_task = asyncio.create_task(_config_watcher())
    logger.info("初始化完成，服务就绪")
    yield
    # 关闭时清理：取消配置热重载后台任务
    app.state.config_reload_task.cancel()
    try:
        await app.state.config_reload_task
    except asyncio.CancelledError:
        pass
    # 释放 PipelineService 持有的 AsyncOpenAI 客户端（HTTP 连接池），
    # 置空全局引用以便 FAISS / BM25 索引等资源随 gc 释放，避免连接泄漏与句柄残留
    pipeline_service = getattr(app.state, "pipeline_service", None)
    if pipeline_service is not None:
        try:
            await pipeline_service.aclose()
        except Exception:
            logger.exception("PipelineService.aclose 失败")
    app.state.pipeline_service = None
    app.state.storage = None
    # 释放 Redis 连接池（embedding 缓存 + 限流）
    try:
        await cache.close_redis()
    except Exception:
        logger.exception("Redis 连接释放失败")
    # 显式回收：触发 FAISS 索引、BM25 索引等全局对象的 __del__，释放底层资源
    gc.collect()
    logger.info("服务已停止，全局资源已释放")


async def _config_watcher(period: float = 10.0) -> None:
    """后台协程：每 period 秒检测配置文件变更并热重载。

    配置重载失败不影响主服务，异常被捕获后等待下一周期继续检测。
    """
    while True:
        try:
            await asyncio.sleep(period)
            reloaded = await maybe_reload_config()
            if reloaded:
                logger.info("配置文件已变更，触发热重载")
        except asyncio.CancelledError:
            # 应用关闭时退出循环
            break
        except Exception:
            # 配置重载失败不应影响主服务
            logger.exception("配置热重载失败，等待下一周期")
            await asyncio.sleep(period)


def create_app() -> FastAPI:
    """创建并返回 FastAPI 应用实例，配置 CORS、注册 lifespan 与路由"""
    application = FastAPI(
        title="企业知识库 RAG 问答服务",
        description="基于 RAG 的企业研报问答 API：PDF 上传入库、SSE 流式问答、会话历史查询",
        version="1.0.0",
        lifespan=lifespan,
    )
    # 全局 CORS：需在注册路由前挂载，确保所有路由（含 SSE /chat）都经过 CORS 处理
    _add_cors_middleware(application)
    # 请求日志中间件（CORS 之后，路由之前）：记录请求 ID、耗时、状态码
    application.add_middleware(RequestLoggingMiddleware)
    # 全局错误处理中间件（最外层）：捕获未处理异常，返回标准错误 JSON，不泄露堆栈
    application.add_middleware(ErrorHandlingMiddleware)
    application.include_router(router)
    return application


# 模块级别应用实例，供 uvicorn 直接引用
app = create_app()


def run() -> None:
    """以 uvicorn 启动服务（命令行入口）"""
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":  # pragma: no cover
    run()
