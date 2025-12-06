import os
import sys
import unittest
from typing import List, Dict, Any, Optional
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from state import AnyMessage, Document, AgentState, add_message

class TestAgentState(unittest.TestCase):
    def setUp(self):
        self.initial_messages: List[AnyMessage] = [
            {"role": "user", "content": "我是一個滑雪新手，我想明年初排去滑雪，有推薦日本的雪場嗎？"}
        ]
        self.new_message: AnyMessage = {"role": "assistant", "content": "請問您要玩幾天"}
        self.user_preferences: Dict[str, Any] = {"budget": "中等", "travel_dates": "明年一月", "activity_level": "初學者"}
        self.retrieved_docs: List[Document] = [
            {"title": "日本滑雪場推薦", "content": "日本有許多優質的滑雪場，如北海道的二世古、長野的白馬等，適合初學者。", "source": "travel_blog_123"},
            {"title": "滑雪裝備指南", "content": "初學者建議租借滑雪裝備，避免購買過多不必要的用品。", "source": "ski_gear_guide"}
        ]
        self.query: Optional[str] = "推薦適合初學者的日本滑雪場"

    def test_add_message(self):
        updated = add_message(self.initial_messages, self.new_message)
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[-1]["role"], "assistant")
        self.assertEqual(updated[-1]["content"], "請問您要玩幾天")

    def test_agent_state_structure(self):
        state: AgentState = {
            "messages": [self.initial_messages[0], self.new_message],
            "user_preferences": self.user_preferences,
            "retrieved_docs": self.retrieved_docs,
            "query": self.query
        }
        self.assertIsInstance(state["messages"], list)
        self.assertIsInstance(state["user_preferences"], dict)
        self.assertIsInstance(state["retrieved_docs"], list)
        self.assertIsInstance(state["query"], str)
        self.assertEqual(state["messages"][0]["role"], "user")
        self.assertEqual(state["retrieved_docs"][0]["title"], "日本滑雪場推薦")

    def test_empty_query(self):
        state: AgentState = {
            "messages": [],
            "user_preferences": {},
            "retrieved_docs": [],
            "query": None
        }
        self.assertIsNone(state["query"])
        self.assertEqual(state["messages"], [])
        self.assertEqual(state["retrieved_docs"], [])

if __name__ == "__main__":
    unittest.main()