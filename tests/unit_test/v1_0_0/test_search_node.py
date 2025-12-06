import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import pytest
from nodes import call_search_tool

def test_call_search_tool(monkeypatch):
    # 模擬 search_attractions 回傳
    dummy_results = [
        {"title": "東京塔", "description": "地標", "url": "https://tokyotower.jp", "score": 4.8},
        {"title": "淺草寺", "description": "古寺", "url": "https://sensoji.jp", "score": 4.7}
    ]
    def mock_search_attractions(query, location, api_key):
        assert query == "東京景點"
        assert location == "東京"
        assert api_key == "dummy_key"
        return dummy_results

    monkeypatch.setattr("tools.search_attractions", mock_search_attractions)

    state = {
        "query": "東京景點",
        "user_preferences": {"location": "東京"},
        "tool_results": {}
    }
    api_key = "dummy_key"
    state = call_search_tool(state, api_key)
    assert "search_results" in state["tool_results"]
    assert state["tool_results"]["search_results"]["result"] == dummy_results