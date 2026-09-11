# app/db.py
"""SQLite 持久化存储：对话历史与会话管理。

基于 aiosqlite 实现异步 IO，避免阻塞事件循环。
表设计：
- sessions: 会话元数据
- chat_history: 问答历史记录（按 session_id 分区，支持分页）
  含完整字段：question / answer / step_by_step_analysis / reasoning_summary /
  relevant_pages(JSON) / references(JSON) / elapsed_seconds / created_at
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import aiosqlite

# 数据库文件路径：优先读环境变量 KB_DB_PATH，默认放在项目根 data/chat_history.db
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chat_history.db"
DB_PATH = Path(os.getenv("KB_DB_PATH", str(_DEFAULT_DB_PATH)))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite 并发写优化：WAL 模式提升读写并发，synchronous=NORMAL 平衡安全与性能
# foreign_keys=ON 启用外键约束，使 ON DELETE CASCADE 生效，保证数据完整性
PRAGMAS = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA mmap_size=30000000000;
PRAGMA foreign_keys=ON;
"""


async def init_db(db_path: Path = DB_PATH) -> None:
    """初始化数据库表结构（幂等，可安全重复调用）。"""
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(PRAGMAS)
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                step_by_step_analysis TEXT DEFAULT '',
                reasoning_summary TEXT DEFAULT '',
                relevant_pages TEXT DEFAULT '[]',
                references_json TEXT DEFAULT '[]',
                elapsed_seconds REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                retry_metadata_json TEXT DEFAULT '{}',
                confidence_json TEXT DEFAULT '{}',
                forced_exit INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
            """
        )
        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_history_session
            ON chat_history(session_id, created_at DESC)
            """
        )

        # 旧库迁移：CREATE TABLE IF NOT EXISTS 不会为已存在的旧表补列，
        # 这里按现有列名动态补加缺失列（幂等，列已存在时跳过），避免历史库
        # 因缺少 retry_metadata_json / confidence_json / forced_exit 列而写入失败。
        cursor = await db.execute("PRAGMA table_info(chat_history)")
        existing_cols = {row[1] for row in await cursor.fetchall()}
        required_cols = {
            "retry_metadata_json": "TEXT DEFAULT '{}'",
            "confidence_json": "TEXT DEFAULT '{}'",
            "forced_exit": "INTEGER DEFAULT 0",
        }
        for col_name, col_def in required_cols.items():
            if col_name not in existing_cols:
                await db.execute(
                    f"ALTER TABLE chat_history ADD COLUMN {col_name} {col_def}"
                )

        await db.commit()


@asynccontextmanager
async def get_db(db_path: Path = DB_PATH) -> AsyncGenerator[aiosqlite.Connection, None]:
    """异步数据库连接上下文管理器：每次操作打开连接、设置行工厂、执行 PRAGMA。"""
    async with aiosqlite.connect(db_path) as db:
        # 使用 Row 工厂，支持按列名访问
        db.row_factory = aiosqlite.Row
        await db.executescript(PRAGMAS)
        yield db


class SQLiteSessionStorage:
    """基于 SQLite 的持久化会话存储，协程安全。

    实现 app.storage.SessionStorage 协议，可与 InMemorySessionStorage 无缝替换。
    每个方法独立获取连接（连接即用即释），无长期占用连接，天然支持并发。
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path

    async def get_or_create_session(self, session_id: str) -> dict:
        """获取或创建会话：不存在则插入，存在则更新 updated_at。"""
        now = datetime.now(timezone.utc).isoformat()
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                await db.execute(
                    "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (session_id, now, now),
                )
                await db.commit()
                return {"session_id": session_id, "history": []}
            await db.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            await db.commit()
            return {"session_id": session_id, "history": []}

    async def get_history(self, session_id: str, limit: int = 100) -> List[dict]:
        """获取指定会话的历史问答记录，按时间正序返回（最新 limit 条）。"""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT question, answer, step_by_step_analysis, reasoning_summary,
                       relevant_pages, references_json, elapsed_seconds, created_at,
                       retry_metadata_json, confidence_json, forced_exit
                FROM chat_history
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
            rows = await cursor.fetchall()
            # 倒序查询后反转为正序，便于前端按时间线展示
            return [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "step_by_step_analysis": r["step_by_step_analysis"],
                    "reasoning_summary": r["reasoning_summary"],
                    "relevant_pages": json.loads(r["relevant_pages"]),
                    "references": json.loads(r["references_json"]),
                    "elapsed_seconds": r["elapsed_seconds"],
                    "created_at": r["created_at"],
                    "retry_metadata": json.loads(r["retry_metadata_json"]) if r["retry_metadata_json"] else None,
                    "confidence": json.loads(r["confidence_json"]) if r["confidence_json"] else None,
                    "forced_exit": bool(r["forced_exit"]),
                }
                for r in reversed(rows)
            ]

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
        """追加一条问答记录，并更新对应会话的 updated_at（若会话不存在则创建）。"""
        now = datetime.now(timezone.utc).isoformat()
        async with get_db(self.db_path) as db:
            # 幂等保证会话存在
            cursor = await db.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
            )
            if await cursor.fetchone() is None:
                await db.execute(
                    "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (session_id, now, now),
                )
            await db.execute(
                """
                INSERT INTO chat_history
                (session_id, question, answer, step_by_step_analysis, reasoning_summary,
                 relevant_pages, references_json, elapsed_seconds, created_at,
                 retry_metadata_json, confidence_json, forced_exit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    question,
                    answer,
                    step_by_step_analysis,
                    reasoning_summary,
                    json.dumps(relevant_pages or [], ensure_ascii=False),
                    json.dumps(references or [], ensure_ascii=False),
                    elapsed_seconds,
                    now,
                    json.dumps(retry_metadata or {}, ensure_ascii=False),
                    json.dumps(confidence or {}, ensure_ascii=False),
                    1 if forced_exit else 0,
                ),
            )
            await db.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            await db.commit()

    async def delete_session(self, session_id: str) -> bool:
        """删除会话及其全部历史记录。

        先显式清理 chat_history，再删除 sessions 行；同时依赖外键
        ON DELETE CASCADE 兜底，双重保证不留孤儿历史记录。
        """
        async with get_db(self.db_path) as db:
            # 显式删除该会话的全部历史记录
            await db.execute(
                "DELETE FROM chat_history WHERE session_id = ?", (session_id,)
            )
            cursor = await db.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def list_sessions(self, limit: int = 100, offset: int = 0) -> List[dict]:
        """列出会话元数据，按最近活跃倒序，支持分页。"""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT s.session_id, s.created_at, s.updated_at, s.metadata,
                       (SELECT COUNT(*) FROM chat_history h WHERE h.session_id = s.session_id) AS message_count
                FROM sessions s
                ORDER BY s.updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
