import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import unittest
from nodes import rag_retrieval_node

class TestRAGNode(unittest.TestCase):
    def test_normal_query(self):
        state = {"messages": [{"role": "user", "content": "推薦台北景點"}]}
        result = rag_retrieval_node(state)
        # print("正常查詢結果：", result)
        self.assertIsInstance(result["retrieved_docs"], list)
        self.assertGreaterEqual(len(result["retrieved_docs"]), 1)

    def test_tokyo_query(self):
        state = {"messages": [{"role": "user", "content": "東京有什麼好玩"}]}
        result = rag_retrieval_node(state)
        # print("東京查詢結果：", result)
        self.assertIsInstance(result["retrieved_docs"], list)
        self.assertGreaterEqual(len(result["retrieved_docs"]), 1)

    def test_budget_query(self):
        state = {"messages": [{"role": "user", "content": "怎麼規劃預算"}]}
        result = rag_retrieval_node(state)
        # print("預算查詢結果：", result)
        self.assertIsInstance(result["retrieved_docs"], list)
        self.assertGreaterEqual(len(result["retrieved_docs"]), 1)

    def test_empty_query(self):
        state = {"messages": []}
        result = rag_retrieval_node(state)
        # print("空查詢結果：", result)
        self.assertEqual(result["retrieved_docs"], [])

    def test_no_result_query(self):
        state = {"messages": [{"role": "user", "content": "火星"}]}
        result = rag_retrieval_node(state)
        # print("無相關結果查詢：", result)
        self.assertIsInstance(result["retrieved_docs"], list)
        # 可根據實際資料庫內容調整
        self.assertTrue(len(result["retrieved_docs"]) == 0 or len(result["retrieved_docs"]) < 3)

    def test_error_handling(self):
        # 模擬 retriever 不存在
        import nodes
        old_vector_store = getattr(nodes, "vector_store", None)
        nodes.vector_store = None
        state = {"messages": [{"role": "user", "content": "推薦台北景點"}]}
        result = rag_retrieval_node(state)
        # print("錯誤處理測試：", result)
        self.assertEqual(result["retrieved_docs"], [])
        nodes.vector_store = old_vector_store

if __name__ == "__main__":
    unittest.main()