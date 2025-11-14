import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from graph import create_graph
from state import AgentState

def print_result(success: bool, msg: str):
    mark = "✅" if success else "❌"
    print(f"{mark} {msg}")

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.app = create_graph()
        self.thread_id = "test-session"
        self.base_state = {
            "messages": [],
            "user_preferences": {},
            "retrieved_docs": [],
            "query": ""
        }

    def test_basic_greeting(self):
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "你好"})
        state["query"] = "你好"
        try:
            result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            msgs = [m for m in result["messages"] if m["role"] == "assistant"]
            self.assertTrue(msgs, "應該有 AI 回應")
            print_result(True, "基本問候測試通過")
        except Exception as e:
            print_result(False, f"基本問候測試失敗: {e}")
            raise

    def test_rag_taipei(self):
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "推薦台北景點"})
        state["query"] = "推薦台北景點"
        try:
            result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            msgs = [m for m in result["messages"] if m["role"] == "assistant"]
            self.assertTrue(msgs, "應該有 AI 回應")
            content = msgs[-1]["content"]
            self.assertIn("台北", content, "回應應包含台北相關資訊")
            print_result(True, "台北景點 RAG 測試通過")
        except Exception as e:
            print_result(False, f"台北景點 RAG 測試失敗: {e}")
            raise

    def test_rag_tokyo(self):
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "東京有什麼好玩"})
        state["query"] = "東京有什麼好玩"
        try:
            result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            msgs = [m for m in result["messages"] if m["role"] == "assistant"]
            self.assertTrue(msgs, "應該有 AI 回應")
            content = msgs[-1]["content"]
            self.assertIn("東京", content, "回應應包含東京相關資訊")
            print_result(True, "東京景點 RAG 測試通過")
        except Exception as e:
            print_result(False, f"東京景點 RAG 測試失敗: {e}")
            raise

    def test_budget(self):
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "怎麼規劃預算"})
        state["query"] = "怎麼規劃預算"
        try:
            result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            msgs = [m for m in result["messages"] if m["role"] == "assistant"]
            self.assertTrue(msgs, "應該有 AI 回應")
            content = msgs[-1]["content"]
            self.assertIn("預算", content, "回應應包含預算相關資訊")
            print_result(True, "預算規劃 RAG 測試通過")
        except Exception as e:
            print_result(False, f"預算規劃 RAG 測試失敗: {e}")
            raise

    def test_no_related(self):
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "火星旅遊"})
        state["query"] = "火星旅遊"
        try:
            result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            msgs = [m for m in result["messages"] if m["role"] == "assistant"]
            self.assertTrue(msgs, "應該有 AI 回應")
            print_result(True, "無相關文件降級處理測試通過")
        except Exception as e:
            print_result(False, f"無相關文件降級處理測試失敗: {e}")
            raise

    def test_multi_turn(self):
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "你好"})
        state["query"] = "你好"
        result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
        state = result
        state["messages"].append({"role": "user", "content": "推薦台北景點"})
        state["query"] = "推薦台北景點"
        try:
            result2 = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            msgs = [m for m in result2["messages"] if m["role"] == "assistant"]
            self.assertTrue(msgs, "應該有 AI 回應")
            self.assertGreaterEqual(len(result2["messages"]), 3, "應該有多輪對話記憶")
            print_result(True, "多輪對話記憶測試通過")
        except Exception as e:
            print_result(False, f"多輪對話記憶測試失敗: {e}")
            raise

if __name__ == "__main__":
    unittest.main()