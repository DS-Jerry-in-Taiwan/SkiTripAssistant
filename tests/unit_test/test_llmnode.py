import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import unittest
from nodes import call_model_node, format_retrieval_node

class TestLLMNode(unittest.TestCase):
    def setUp(self):
        # 範例文件（模擬 Document 物件）
        self.doc_short = type("Doc", (), {"metadata": {"title": "台北景點"}, "page_content": "故宮、101、士林夜市"})()
        self.doc_long = type("Doc", (), {"metadata": {"title": "預算指南"}, "page_content": "A"*500})()
        self.state_base = {
            "messages": [{"role": "user", "content": "推薦台北景點"}],
            "retrieved_docs": [self.doc_short],
        }

    def test_llm_reads_retrieved_docs(self):
        state = self.state_base.copy()
        result_state = call_model_node(state)
        # 檢查 assistant 回應已加入
        self.assertTrue(any(msg["role"] == "assistant" for msg in result_state["messages"]))

    def test_prompt_injects_docs(self):
        state = self.state_base.copy()
        prompt = format_retrieval_node(state)
        self.assertIn("台北景點", prompt)
        self.assertIn("故宮", prompt)

    def test_llm_no_docs(self):
        state = self.state_base.copy()
        state["retrieved_docs"] = []
        result_state = call_model_node(state)
        # 檢查 assistant 回應已加入且無錯誤
        self.assertTrue(any(msg["role"] == "assistant" for msg in result_state["messages"]))

    def test_long_doc_truncation(self):
        state = self.state_base.copy()
        state["retrieved_docs"] = [self.doc_long]
        prompt = format_retrieval_node(state)
        # 檢查是否有截斷提示
        self.assertIn("truncated", prompt)
        self.assertLessEqual(len(prompt), 1200)
        
    def test_extremely_long_doc(self):
        # 建立超長文件（超過 5000 字）
        long_text = "B" * 6000
        doc_extreme = type("Doc", (), {"metadata": {"title": "極長文件"}, "page_content": long_text})()
        state = self.state_base.copy()
        state["retrieved_docs"] = [doc_extreme]
        prompt = format_retrieval_node(state)
        # 檢查是否有截斷提示
        self.assertIn("truncated", prompt)
        self.assertLessEqual(len(prompt), 1200)

if __name__ == "__main__":
    unittest.main()