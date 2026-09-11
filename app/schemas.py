# Pydantic 数据模型定义
import re
from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator

# 会话 ID 仅允许字母、数字、下划线、连字符
_SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
# 问题中必须出现被英文双引号包裹的公司名（公司名长度 2-50）
_QUOTED_COMPANY_PATTERN = re.compile(r'"[^"]{2,50}"')


class ChatRequest(BaseModel):
    """POST /chat 请求体"""
    session_id: str = Field(..., description="会话 ID，用于关联同一会话的多轮问答，长度 3-64，仅允许字母、数字、下划线与连字符")
    question: str = Field(..., description="用户问题，需用英文双引号包裹公司名，长度 10-500")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        """校验会话 ID：长度 3-64，仅允许 a-zA-Z0-9_-"""
        if len(v) < 3 or len(v) > 64:
            raise ValueError("会话 ID 长度需在 3-64 个字符之间")
        if not _SESSION_ID_PATTERN.fullmatch(v):
            raise ValueError("会话 ID 仅允许字母、数字、下划线与连字符")
        return v

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """校验问题：长度 10-500，且必须包含被英文双引号包裹的公司名"""
        if len(v) < 10 or len(v) > 500:
            raise ValueError("问题长度需在 10-500 个字符之间")
        if not _QUOTED_COMPANY_PATTERN.search(v):
            raise ValueError("问题中必须使用英文双引号包裹公司名")
        return v


class ConfidenceScore(BaseModel):
    """置信度评估结果（低置信度自动查询改写重试）"""
    overall: float = 0.0
    retrieval_confidence: float = 0.0
    faithfulness: float = 0.0
    completeness: float = 0.0
    critique: str = ""
    should_retry: bool = False


class RetryMetadata(BaseModel):
    """重试循环元数据（每轮 query / confidence / critique / strategy）"""
    rounds: List[Any] = []
    forced_exit: bool = False
    total_attempts: int = 0
    final_confidence: Any = None


class ChatResponse(BaseModel):
    """POST /chat 非流式（stream=false）响应体：完整结构化答案"""
    session_id: str
    question: str
    answer: str = ""
    step_by_step_analysis: str = ""
    reasoning_summary: str = ""
    relevant_pages: List[Any] = []
    references: List[Any] = []
    elapsed_seconds: float = 0.0
    confidence: Optional[ConfidenceScore] = None
    retry_metadata: Optional[RetryMetadata] = None
    forced_exit: bool = False


class QARecord(BaseModel):
    """单条问答历史记录"""
    question: str
    answer: str
    step_by_step_analysis: str = ""
    reasoning_summary: str = ""
    relevant_pages: List[Any] = []
    references: List[Any] = []
    elapsed_seconds: float = 0.0
    created_at: str = ""
    confidence: Optional[Any] = None
    retry_metadata: Optional[Any] = None
    forced_exit: bool = False


class HistoryResponse(BaseModel):
    """GET /history/{session_id} 响应体"""
    session_id: str
    total: int
    records: List[QARecord]


class UploadResponse(BaseModel):
    """POST /upload 响应体（同步入库旧契约，保留供 ingest_pdf 直连流程使用）"""
    status: str
    file_name: str
    company_name: str
    sha1: str = ""
    message: str = ""
    elapsed_seconds: float = 0.0


class TaskSubmitResponse(BaseModel):
    """POST /upload 异步化响应体：受理成功即返回 Celery 任务 ID"""
    task_id: str
    status: str = "pending"


class TaskStatusResponse(BaseModel):
    """GET /tasks/{task_id} 响应体：Celery 任务状态查询"""
    task_id: str
    status: str = "pending"
    detail: Optional[Any] = None


class HealthResponse(BaseModel):
    """GET /health 响应体"""
    status: str


class ChatDoneEvent(BaseModel):
    """SSE done 事件携带的完整结果信息"""
    session_id: str
    question: str
    answer: str
    relevant_pages: List[Any] = []
    references: List[Any] = []
    elapsed_seconds: float = 0.0
