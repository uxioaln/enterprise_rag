# 单元测试：InMemoryStorage 会话存储的 CRUD 操作
import pytest

from app.storage import InMemoryStorage


class TestInMemoryStorageCRUD:
    """测试内存存储的增删查基本操作"""

    def test_add_and_get_single_record(self):
        """添加一条记录后应能查到"""
        store = InMemoryStorage()
        record = {"question": "问题A", "answer": "答案A"}
        store.add_record("session-1", record)

        history = store.get_history("session-1")
        assert len(history) == 1
        assert history[0]["question"] == "问题A"
        assert history[0]["answer"] == "答案A"

    def test_add_multiple_records_same_session(self):
        """同一会话添加多条记录，应按顺序返回"""
        store = InMemoryStorage()
        store.add_record("session-1", {"question": "Q1", "answer": "A1"})
        store.add_record("session-1", {"question": "Q2", "answer": "A2"})
        store.add_record("session-1", {"question": "Q3", "answer": "A3"})

        history = store.get_history("session-1")
        assert len(history) == 3
        assert history[0]["question"] == "Q1"
        assert history[1]["question"] == "Q2"
        assert history[2]["question"] == "Q3"

    def test_get_history_nonexistent_session(self):
        """查询不存在的会话应返回空列表"""
        store = InMemoryStorage()
        history = store.get_history("nonexistent")
        assert history == []

    def test_get_history_returns_copy(self):
        """get_history 返回的列表应是副本，修改不影响内部数据"""
        store = InMemoryStorage()
        store.add_record("session-1", {"question": "Q1", "answer": "A1"})

        history = store.get_history("session-1")
        history.clear()

        # 内部数据不应受影响
        assert len(store.get_history("session-1")) == 1

    def test_multiple_sessions_isolated(self):
        """多个会话之间数据应隔离"""
        store = InMemoryStorage()
        store.add_record("session-A", {"question": "QA", "answer": "AA"})
        store.add_record("session-B", {"question": "QB", "answer": "AB"})
        store.add_record("session-B", {"question": "QB2", "answer": "AB2"})

        assert len(store.get_history("session-A")) == 1
        assert len(store.get_history("session-B")) == 2

    def test_list_sessions(self):
        """list_sessions 应返回所有已创建的会话 ID"""
        store = InMemoryStorage()
        store.add_record("session-A", {"q": "1"})
        store.add_record("session-B", {"q": "2"})

        sessions = store.list_sessions()
        assert set(sessions) == {"session-A", "session-B"}

    def test_build_record_structure(self):
        """build_record 应正确构建包含完整字段的历史记录"""
        answer_dict = {
            "final_answer": "最终答案",
            "step_by_step_analysis": "分步分析",
            "reasoning_summary": "推理摘要",
            "relevant_pages": [{"file_name": "test.pdf", "page": 1}],
            "references": [{"pdf_file_name": "test.pdf", "page_index": 1}],
        }
        record = InMemoryStorage.build_record("测试问题", answer_dict, elapsed_seconds=1.5)

        assert record["question"] == "测试问题"
        assert record["answer"] == "最终答案"
        assert record["step_by_step_analysis"] == "分步分析"
        assert record["reasoning_summary"] == "推理摘要"
        assert record["relevant_pages"] == [{"file_name": "test.pdf", "page": 1}]
        assert record["references"] == [{"pdf_file_name": "test.pdf", "page_index": 1}]
        assert record["elapsed_seconds"] == 1.5
        assert "created_at" in record
