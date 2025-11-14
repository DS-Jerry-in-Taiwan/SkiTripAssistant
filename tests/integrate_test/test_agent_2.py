import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from graph import create_graph

def print_result(success: bool, msg: str):
    mark = "✅" if success else "❌"
    print(f"{mark} {msg}")

class TestRAGIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_graph()
        cls.thread_id = "rag-integration-test"
        cls.base_state = {
            "messages": [],
            "user_preferences": {},
            "retrieved_docs": [],
            "query": ""
        }

    def test_taipei_attractions(self):
        """知識庫內容驗證：台北景點"""
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "推薦台北景點"})
        state["query"] = "推薦台北景點"
        try:
            result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            msgs = [m for m in result["messages"] if m["role"] == "assistant"]
            self.assertTrue(msgs, "應該有 AI 回應")
            content = msgs[-1]["content"]
            self.assertTrue(any(x in content for x in ["故宮", "101", "夜市"]), "回應應包含台北文件庫內容")
            print_result(True, "台北景點知識庫驗證通過")
        except Exception as e:
            print_result(False, f"台北景點知識庫驗證失敗: {e}")
            raise

    def test_tokyo_attractions(self):
        """知識庫內容驗證：東京景點"""
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "東京有什麼好玩的"})
        state["query"] = "東京有什麼好玩的"
        try:
            result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            msgs = [m for m in result["messages"] if m["role"] == "assistant"]
            self.assertTrue(msgs, "應該有 AI 回應")
            content = msgs[-1]["content"]
            self.assertTrue(any(x in content for x in ["淺草寺", "晴空塔"]), "回應應包含東京文件庫內容")
            print_result(True, "東京景點知識庫驗證通過")
        except Exception as e:
            print_result(False, f"東京景點知識庫驗證失敗: {e}")
            raise

    def test_budget_guide(self):
        """知識庫內容驗證：預算規劃"""
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "旅遊預算怎麼規劃"})
        state["query"] = "旅遊預算怎麼規劃"
        try:
            result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            msgs = [m for m in result["messages"] if m["role"] == "assistant"]
            self.assertTrue(msgs, "應該有 AI 回應")
            content = msgs[-1]["content"]
            self.assertTrue(any(x in content for x in ["經濟", "中等", "豪華"]), "回應應參考預算指南")
            print_result(True, "預算規劃知識庫驗證通過")
        except Exception as e:
            print_result(False, f"預算規劃知識庫驗證失敗: {e}")
            raise

    def test_retrieval_accuracy(self):
        """檢索準確性驗證"""
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "推薦台北景點"})
        state["query"] = "推薦台北景點"
        try:
            result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            docs = result.get("retrieved_docs", [])
            self.assertTrue(docs, "應該有檢索到文件")
            self.assertLessEqual(len(docs), 3, "檢索文件數量應最多3個")
            doc_titles = [getattr(doc, "metadata", {}).get("title", "") for doc in docs]
            self.assertTrue(any("台北" in title for title in doc_titles), "檢索到的文件應與查詢相關")
            print_result(True, "檢索準確性驗證通過")
        except Exception as e:
            print_result(False, f"檢索準確性驗證失敗: {e}")
            raise

    def test_fallback(self):
        """降級處理驗證：無相關文件"""
        state = self.base_state.copy()
        state["messages"].append({"role": "user", "content": "巴黎有什麼景點"})
        state["query"] = "巴黎有什麼景點"
        try:
            result = self.app.invoke(input=state, config={"configurable": {"thread_id": self.thread_id}})
            docs = result.get("retrieved_docs", [])
            msgs = [m for m in result["messages"] if m["role"] == "assistant"]
            self.assertTrue(msgs, "應該有 AI 回應")
            content = msgs[-1]["content"]
            self.assertTrue("巴黎" in content, "回應應包含巴黎")
            self.assertTrue("沒有相關資料" in content or "通用知識" in content or len(docs) == 0, "應有降級處理提示")
            print_result(True, "降級處理驗證通過")
        except Exception as e:
            print_result(False, f"降級處理驗證失敗: {e}")
            raise

if __name__ == "__main__":
    unittest.main()