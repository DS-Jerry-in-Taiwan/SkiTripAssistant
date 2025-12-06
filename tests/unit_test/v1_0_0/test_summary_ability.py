import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from nodes import generate_conversation_summary

import pytest
from nodes import generate_conversation_summary

def mock_state(messages, user_preferences=None):
    return {
        "messages": messages,
        "user_preferences": user_preferences or {}
    }

def test_conversation_summary_basic():
    messages = [
        {"role": "user", "content": "我想去日本滑雪，有推薦的地方嗎？"},
        {"role": "assistant", "content": "北海道二世古很適合滑雪。"},
        {"role": "user", "content": "預算大約五萬元，想一月出發。"},
        {"role": "assistant", "content": "一月是旺季，建議提前訂房。"}
    ]
    user_preferences = {"budget": "50000", "travel_dates": "一月"}
    state = mock_state(messages, user_preferences)
    summary = generate_conversation_summary(state)
    assert "日本" in summary
    assert "滑雪" in summary
    assert "預算" in summary
    assert "一月" in summary

def test_conversation_summary_multi_turn():
    messages = [
        {"role": "user", "content": "我想去歐洲自助旅行。"},
        {"role": "assistant", "content": "請問有特定國家或城市嗎？"},
        {"role": "user", "content": "想去法國和德國，預算八萬元。"},
        {"role": "assistant", "content": "建議巴黎和柏林都可以安排。"},
        {"role": "user", "content": "我喜歡美食和博物館。"}
    ]
    user_preferences = {"budget": "80000", "destination_preferences": "法國, 德國"}
    state = mock_state(messages, user_preferences)
    summary = generate_conversation_summary(state)
    assert "法國" in summary
    assert "德國" in summary
    assert "美食" in summary
    assert "博物館" in summary

# 更多測試可根據實際需求擴充