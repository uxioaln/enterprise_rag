# app/middleware.py
"""FastAPI 中间件：请求日志记录与全局异常统一处理。

架构位置（从外到内）：
    ErrorHandlingMiddleware (最外层)
    -> RequestLoggingMiddleware
    -> CORSMiddleware
    -> 路由层
"""
from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


# --------------------------------------------------------------------------- #
# 1. 请求日志中间件
# --------------------------------------------------------------------------- #
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个 HTTP 请求的详细日志，支持结构化 JSON 输出。

    记录字段：
    - request_id: 唯一追踪 ID
    - method, path, query
    - client, user_agent
    - status
    - duration_ms
    - error（若异常）
    """

    # 跳过日志记录的路径：健康检查与文档端点，避免噪音
    _SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(
        self,
        app,
        logger: Optional[logging.Logger] = None,
        skip_paths: Optional[set[str]] = None,
    ):
        super().__init__(app)
        self.skip_paths = skip_paths or self._SKIP_PATHS
        # 若未传入 logger，使用标准库 logging
        self.logger = logger or logging.getLogger("kb.access")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.skip_paths:
            return await call_next(request)

        # 生成唯一追踪 ID 并挂到 request.state，供下游与错误中间件复用
        request_id = str(uuid.uuid4())[:12]
        request.state.request_id = request_id

        start = time.perf_counter()
        client = request.client.host if request.client else "-"
        ua = request.headers.get("user-agent", "-")

        try:
            response = await call_next(request)
        except Exception as exc:
            # 记录异常后重新抛出，由外层 ErrorHandlingMiddleware 兜底处理
            self._log(request, request_id, start, 500, client, ua, error=str(exc))
            raise
        else:
            duration = (time.perf_counter() - start) * 1000
            self._log(
                request,
                request_id,
                start,
                response.status_code,
                client,
                ua,
                duration=duration,
            )
            # 注入追踪 ID 到响应头，便于前端/运维关联日志
            response.headers["X-Request-ID"] = request_id
            return response

    def _log(
        self,
        request: Request,
        request_id: str,
        start: float,
        status: int,
        client: str,
        ua: str,
        duration: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        duration_ms = duration if duration is not None else (time.perf_counter() - start) * 1000
        log_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params),
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "client": client,
            "user_agent": ua,
        }
        if error:
            log_data["error"] = error

        # 根据状态码选择日志级别，便于告警与过滤
        if status >= 500:
            self.logger.error(json.dumps(log_data, ensure_ascii=False))
        elif status >= 400:
            self.logger.warning(json.dumps(log_data, ensure_ascii=False))
        else:
            self.logger.info(json.dumps(log_data, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# 2. 全局错误处理中间件
# --------------------------------------------------------------------------- #
class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """捕获所有未处理异常，返回标准错误响应，避免堆栈信息泄露。

    错误响应格式：
    {
        "error": {
            "type": "internal" | "validation" | "business" | ...,
            "message": "用户友好提示",
            "request_id": "xxx"
        }
    }
    """

    def __init__(self, app, logger: Optional[logging.Logger] = None):
        super().__init__(app)
        self.logger = logger or logging.getLogger("kb.error")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            # 复用请求日志中间件注入的 request_id，便于跨日志关联
            request_id = getattr(request.state, "request_id", "unknown")
            # 完整堆栈仅记录到服务端日志，绝不返回给客户端
            self.logger.error(
                json.dumps(
                    {
                        "request_id": request_id,
                        "type": "unhandled_exception",
                        "exception": type(exc).__name__,
                        "detail": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                )
            )

            # 根据异常类型映射 HTTP 状态码与错误类型
            status_code, error_type, user_msg = self._classify_exception(exc)

            return JSONResponse(
                status_code=status_code,
                content={
                    "error": {
                        "type": error_type,
                        "message": user_msg,
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )

    def _classify_exception(self, exc: Exception) -> tuple[int, str, str]:
        """将异常分类为 HTTP 状态码和错误类型，向用户暴露最小化信息。"""
        from fastapi.exceptions import RequestValidationError

        name = type(exc).__name__

        # 显式映射已知异常类型
        if isinstance(exc, RequestValidationError):
            return 422, "validation", "请求参数校验失败，请检查输入格式"
        # 自定义限流异常（app.rate_limiter.RateLimitExceeded）：返回 429
        # 延迟导入避免中间件模块加载时引入 redis 依赖
        if name == "RateLimitExceeded":
            try:
                from app.rate_limiter import RateLimitExceeded as _RLE
                if isinstance(exc, _RLE):
                    return 429, "rate_limit", "请求过于频繁，请稍后再试"
            except ImportError:
                pass
        if name in ("RateLimitError", "APITimeoutError", "APIConnectionError"):
            return 503, "model_service", "模型服务暂时不可用，请稍后重试"
        if name == "HTTPException":
            # FastAPI HTTPException：透传其自带的 status_code 与 detail
            code = getattr(exc, "status_code", 500)
            detail = getattr(exc, "detail", "请求处理失败")
            if code == 404:
                return 404, "not_found", str(detail)
            if code == 403:
                return 403, "forbidden", str(detail)
            return code, "business", str(detail)

        # 业务异常（自定义，按命名约定识别）
        if name.endswith("BusinessError"):
            return 400, "business", str(exc)

        # 默认：内部错误，不向客户端暴露任何细节
        return 500, "internal", "系统内部错误，请联系管理员"
