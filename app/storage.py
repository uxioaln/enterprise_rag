# 简单的内存存储类，用于管理 session_id 与对话历史
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol


class InMemoryStorage:
    """基于字典的内存会话存储，线程安全。

    结构：{session_id: [问答记录 dict, ...]}

    说明：此为早期同步实现，保留用于既有单元测试与向后兼容。
    生产与异步服务层请使用 InMemorySessionStorage / SQLiteSessionStorage。
    """

    def __init__(self):
        # 会话历史字典，key 为 session_id，value 为该会话下的问答记录列表
        self._data: Dict[str, List[dict]] = {}
        # 读写锁，保证多线程（run_in_executor 线程池）访问安全
        self._lock = threading.Lock()

    def add_record(self, session_id: str, record: dict) -> None:
        """向指定会话追加一条问答记录"""
        with self._lock:
            self._data.setdefault(session_id, []).append(record)

    def get_history(self, session_id: str) -> List[dict]:
        """获取指定会话的全部问答记录（返回副本，避免外部直接修改内部数据）"""
        with self._lock:
            return list(self._data.get(session_id, []))

    def list_sessions(self) -> List[str]:
        """列出当前所有会话 ID"""
        with self._lock:
            return list(self._data.keys())

    @staticmethod
    def build_record(question: str, answer_dict: dict, elapsed_seconds: float) -> dict:
        """根据 Pipeline 返回的结构化答案构建一条历史记录"""
        return {
            "question": question,
            "answer": answer_dict.get("final_answer", ""),
            "step_by_step_analysis": answer_dict.get("step_by_step_analysis", ""),
            "reasoning_summary": answer_dict.get("reasoning_summary", ""),
            "relevant_pages": answer_dict.get("relevant_pages", []),
            "references": answer_dict.get("references", []),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }


# --------------------------------------------------------------------------- #
# 异步存储协议：内存与 SQLite 实现共同遵守的接口契约
# --------------------------------------------------------------------------- #
class SessionStorage(Protocol):
    """存储后端协议，确保内存与 SQLite 实现接口一致。

    所有方法均为 async：DB 实现使用 aiosqlite 真正异步 IO；
    内存实现虽无 IO，亦保持 async 签名以统一调用方式。
    """

    async def get_or_create_session(self, session_id: str) -> dict: ...

    async def get_history(self, session_id: str, limit: int = 100) -> List[dict]: ...

    async def append_record(
        self,
        session_id: str,
        question: str,
        answer: str,
        step_by_step_analysis: str = "",
        reasoning_summary: str = "",
        relevant_pages: Optional[List[dict]] = None,
        references: Optional[List[dict]] = None,
        elapsed_seconds: float = 0.0,
        confidence: Optional[dict] = None,
        retry_metadata: Optional[dict] = None,
        forced_exit: bool = False,
    ) -> None: ...

    async def delete_session(self, session_id: str) -> bool: ...

    async def list_sessions(self, limit: int = 100, offset: int = 0) -> List[dict]: ...


# --------------------------------------------------------------------------- #
# 异步内存实现（开发/测试默认后端）
# --------------------------------------------------------------------------- #
from collections import defaultdict
from threading import Lock


class InMemorySessionStorage:
    """线程/协程安全的异步内存会话存储，实现 SessionStorage 协议。

    与早期同步 InMemoryStorage 接口对齐到异步协议，字段完整保留
    step_by_step_analysis / reasoning_summary，便于与 SQLite 后端无缝切换。
    """

    def __init__(self) -> None:
        # session_id -> 该会话的问答记录列表（dict）
        self._data: dict[str, list] = defaultdict(list)
        self._lock = Lock()

    async def get_or_create_session(self, session_id: str) -> dict:
        with self._lock:
            return {"session_id": session_id, "history": list(self._data.get(session_id, []))}

    async def get_history(self, session_id: str, limit: int = 100) -> List[dict]:
        with self._lock:
            records = self._data.get(session_id, [])[-limit:]
            # 返回副本，避免外部修改内部数据
            return [dict(r) for r in records]

    async def append_record(
        self,
        session_id: str,
        question: str,
        answer: str,
        step_by_step_analysis: str = "",
        reasoning_summary: str = "",
        relevant_pages: Optional[List[dict]] = None,
        references: Optional[List[dict]] = None,
        elapsed_seconds: float = 0.0,
        confidence: Optional[dict] = None,
        retry_metadata: Optional[dict] = None,
        forced_exit: bool = False,
    ) -> None:
        record = {
            "question": question,
            "answer": answer,
            "step_by_step_analysis": step_by_step_analysis,
            "reasoning_summary": reasoning_summary,
            "relevant_pages": relevant_pages or [],
            "references": references or [],
            "elapsed_seconds": round(elapsed_seconds, 2),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "confidence": confidence,
            "retry_metadata": retry_metadata,
            "forced_exit": forced_exit,
        }
        with self._lock:
            self._data[session_id].append(record)

    async def delete_session(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._data
            self._data.pop(session_id, None)
            return existed

    async def list_sessions(self, limit: int = 100, offset: int = 0) -> List[dict]:
        with self._lock:
            items = list(self._data.items())[offset : offset + limit]
            return [
                {
                    "session_id": sid,
                    "created_at": None,
                    "updated_at": None,
                    "message_count": len(hist),
                }
                for sid, hist in items
            ]

# --------------------------------------------------------------------------- #
# 工厂函数：根据环境变量自动选择后端
# --------------------------------------------------------------------------- #
def get_storage() -> SessionStorage:
    """根据环境变量 STORAGE_BACKEND 选择存储后端。

    - sqlite（默认，推荐生产）：持久化，aiosqlite 异步 IO，WAL 模式
    - memory：纯内存，零依赖，用于开发与测试
    """
    backend = os.getenv("STORAGE_BACKEND", "sqlite").lower().strip()
    if backend == "sqlite":
        # 懒导入，避免未使用 SQLite 时引入 aiosqlite
        from app.db import SQLiteSessionStorage

        return SQLiteSessionStorage()
    return InMemorySessionStorage()
