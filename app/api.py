# API 路由定义：upload / tasks / chat / history / health
import asyncio
import time
from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.celery_app import celery_app
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    HistoryResponse,
    TaskStatusResponse,
    TaskSubmitResponse,
)
from app.services import PipelineService, process_chat
from app.storage import SessionStorage
from app.tasks import parse_and_index_pdf
from fastapi.responses import RedirectResponse

router = APIRouter()

# 单次读取上传文件的块大小（1MB）
_UPLOAD_CHUNK_SIZE = 1024 * 1024
# 上传文件大小上限（50MB）
MAX_UPLOAD_SIZE = 50 * 1024 * 1024


# ---------- 依赖注入 ----------

def get_pipeline_service(request: Request) -> PipelineService:
    """从 app.state 获取 PipelineService 实例"""
    return request.app.state.pipeline_service


def get_storage(request: Request) -> SessionStorage:
    """从 app.state 获取会话存储实例（SQLite 或内存，统一异步接口）"""
    return request.app.state.storage


# ---------- 接口 ----------
@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

@router.post("/upload", response_model=TaskSubmitResponse, status_code=202)
async def upload_pdf(
    file: UploadFile = File(..., description="待上传的 PDF 研报文件"),
    company_name: str = Form(..., min_length=2, max_length=50, description="文件所属公司名（必填，长度 2-50），需与提问中的公司名一致"),
    service: PipelineService = Depends(get_pipeline_service),
):
    """上传 PDF 并受理入库：保存文件 -> 登记 subset + OSS 上传 -> 提交 Celery 异步任务。

    接口立即返回 202 与 task_id；解析（MinerU）、分块、向量化入库由后台任务异步执行，
    进度可通过 GET /tasks/{task_id} 查询。
    """
    # 校验文件名与扩展名
    if not file.filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    file_name = Path(file.filename).name
    if not file_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    # 保存到 data/stock_data/pdf_reports/（分块异步读取，写盘放线程池）
    dest_path = service.pdf_reports_dir / file_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_buffer = dest_path.with_suffix(".pdf.uploading")
    total_written = 0
    try:
        with open(tmp_buffer, "wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"文件过大，上限 {MAX_UPLOAD_SIZE // (1024 * 1024)}MB",
                    )
                out.write(chunk)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: tmp_buffer.replace(dest_path))
    except HTTPException:
        tmp_buffer.unlink(missing_ok=True)
        raise
    except Exception as e:
        tmp_buffer.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")
    finally:
        await file.close()

    # 受理前置步骤：登记 subset.csv（取得 sha1）（阻塞操作放线程池）
    try:
        loop = asyncio.get_running_loop()
        info = await loop.run_in_executor(
            None, lambda: service.register_pdf(file_name, company_name)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件登记失败: {e}")

    # 提交 Celery 异步任务：MinerU 解析（签名 URL 直传本地文件）-> 分块 -> 向量化入库
    try:
        task = parse_and_index_pdf.delay(
            company_name, file_name, info["sha1"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"异步任务提交失败: {e}")

    return TaskSubmitResponse(task_id=task.id, status="pending")


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """查询 PDF 入库异步任务状态：pending（排队/执行中）/ success / failure"""
    result = AsyncResult(task_id, app=celery_app)
    state = result.state
    if state == "SUCCESS":
        status = "success"
        detail = result.result if isinstance(result.result, dict) else str(result.result)
    elif state == "FAILURE":
        status = "failure"
        detail = str(result.result)
    else:
        # PENDING / STARTED / RETRY 等统一视为处理中
        status = "pending"
        detail = None
    return TaskStatusResponse(task_id=task_id, status=status, detail=detail)


@router.post("/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    stream: bool = Query(default=True, description="是否流式响应（SSE），false 时返回标准 JSON"),
    service: PipelineService = Depends(get_pipeline_service),
    storage: SessionStorage = Depends(get_storage),
):
    """问答接口：默认 SSE 流式逐句返回，stream=false 时返回标准 JSON（ChatResponse 模型）"""
    if not stream:
        # 非流式：同步获取完整答案后一次性返回标准 JSON（ChatResponse 模型）
        t0 = time.time()
        answer_dict = await service.answer_question(body.question)
        elapsed = time.time() - t0
        # 异步写入会话历史（完整保留推理与引用字段 + 重试元数据）
        retry_meta = answer_dict.get("retry_metadata")
        forced_exit = bool(retry_meta.get("forced_exit", False)) if isinstance(retry_meta, dict) else False
        await storage.append_record(
            body.session_id,
            body.question,
            answer_dict.get("final_answer", ""),
            step_by_step_analysis=answer_dict.get("step_by_step_analysis", ""),
            reasoning_summary=answer_dict.get("reasoning_summary", ""),
            relevant_pages=answer_dict.get("relevant_pages", []),
            references=answer_dict.get("references", []),
            elapsed_seconds=elapsed,
            confidence=answer_dict.get("confidence"),
            retry_metadata=answer_dict.get("retry_metadata"),
            forced_exit=forced_exit,
        )
        return ChatResponse(
            session_id=body.session_id,
            question=body.question,
            answer=answer_dict.get("final_answer", ""),
            step_by_step_analysis=answer_dict.get("step_by_step_analysis", ""),
            reasoning_summary=answer_dict.get("reasoning_summary", ""),
            relevant_pages=answer_dict.get("relevant_pages", []),
            references=answer_dict.get("references", []),
            elapsed_seconds=round(elapsed, 2),
            confidence=answer_dict.get("confidence"),
            retry_metadata=answer_dict.get("retry_metadata"),
            forced_exit=forced_exit,
        )

    # 注入 Request 对象供 process_chat 检测客户端断开，断开后立即取消模型流
    generator = process_chat(
        service, storage, body.session_id, body.question, request=request
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关闭 Nginx 等反向代理的缓冲，保证 SSE 及时下发
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(
    session_id: str,
    storage: SessionStorage = Depends(get_storage),
):
    """获取指定会话的全部历史问答记录，不存在时返回 404"""
    records = await storage.get_history(session_id)
    if not records:
        raise HTTPException(status_code=404, detail=f"未找到会话: {session_id}")
    return HistoryResponse(session_id=session_id, total=len(records), records=records)


@router.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(status="ok")
