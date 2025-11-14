import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import unittest
from graph import create_graph
from state import AgentState

class TestRAGGraph(unittest.TestCase):
    def setUp(self):
        # 建立初始狀態，包含一則 user 訊息
        self.init_state = AgentState(
            messages=[{"role": "user", "content": "推薦台北景點"}],
            user_preferences={},
            retrieved_docs=[],
            query="推薦台北景點"
        )

    def test_graph_compile(self):
        # ✅ Graph 成功編譯無錯誤
        app = create_graph()
        self.assertIsNotNone(app)

    def test_graph_flow_and_state(self):
        # ✅ RAG → LLM 流程正確連接
        app = create_graph()
        # 執行流程，模擬 thread_id
        thread_id = "test_session"
        result_state = app.invoke(self.init_state, config={"configurable": {"thread_id": thread_id}})
        # 檢查狀態在節點間正確傳遞
        self.assertIn("messages", result_state)
        self.assertTrue(any(msg["role"] == "assistant" for msg in result_state["messages"]))
        self.assertIn("retrieved_docs", result_state)
        # 檢查 RAG 檢索結果
        self.assertIsInstance(result_state["retrieved_docs"], list)

    def test_memory_saver(self):
        # ✅ MemorySaver 正常運作
        app = create_graph()
        thread_id = "memory_test"
        state1 = AgentState(messages=[{"role": "user", "content": "你好"}], user_preferences={}, retrieved_docs=[], query="你好")
        state2 = app.invoke(state1, config={"configurable": {"thread_id": thread_id}})
        # 再次呼叫同一 thread_id，應保留對話歷史
        state3 = AgentState(messages=[{"role": "user", "content": "推薦東京景點"}], user_preferences={}, retrieved_docs=[], query="推薦東京景點")
        result_state = app.invoke(state1, config={"configurable": {"thread_id": thread_id}})
        self.assertGreaterEqual(len(result_state["messages"]), 2)
        # 檢查是否有多輪對話記憶

if __name__ == "__main__":
    unittest.main()