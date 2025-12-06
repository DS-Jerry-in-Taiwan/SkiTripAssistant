import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import pytest
import tools

def test_generate_intinerary_llm(monkeypatch):
    # 模擬 openai.ChatCompletion.create 回傳
    class DummyResponse:
        class Choices:
            def __init__(self):
                self.message = type("msg", (), {"content": "Day 1: 東京塔\nDay 2: 淺草寺\nDay 3: 自由活動"})
        choices = [Choices()]
    def mock_chat_completion_create(model, api_key, messages, temperature):
        return DummyResponse()
    
    monkeypatch.setattr("openai.ChatCompletion.create", mock_chat_completion_create)

    attractions = [
        {"title": "東京塔", "description": "地標", "url": "https://tokyotower.jp", "score": 4.8},
        {"title": "淺草寺", "description": "古寺", "url": "https://sensoji.jp", "score": 4.7}
    ]
    user_preferences = {"location": "東京", "days": 3}
    query = "請幫我規劃東京三天行程"
    api_key = "dummy_key"

    result = tools.generate_intinerary(attractions, user_preferences, query, api_key)
    assert "itinerary" in result
    assert "Day 1" in result["itinerary"]
    assert "guide" in result
    assert result["guide"] == ""