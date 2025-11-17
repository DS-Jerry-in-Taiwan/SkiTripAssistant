import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import pytest
from state import AgentState
from graph import create_graph

@pytest.fixture
def initial_state():
    # 初始狀態，模擬一個用戶查詢
    return AgentState(
        messages=[{"role": "user", "content": "我想去東京賞櫻，有推薦景點嗎？"}],
        user_preferences={"location": "東京", "days": 3, "budget": "中等"},
        retrieved_docs=[],
        query="我想去東京賞櫻，有推薦景點嗎？",
        conversation_summary=None,
        tool_results={}
    )

def test_agent_function_calling_and_data_flow(initial_state):
    # 建立 workflow
    app = create_graph()
    # 執行 workflow，模擬一次完整 function calling
    result_state = app.invoke(initial_state)
    
    # 驗證景點搜尋結果
    assert "search_results" in result_state.get("tool_results", {})
    assert isinstance(result_state["tool_results"]["search_results"]["result"], list)
    
    # 驗證行程生成結果
    assert "itinerary" in result_state.get("tool_results", {})
    assert isinstance(result_state["tool_results"]["itinerary"]["result"], str)
    
    # 驗證 LLM 回應已加入 messages
    assistant_msgs = [msg for msg in result_state["messages"] if msg["role"] == "assistant"]
    assert len(assistant_msgs) > 0

    # 驗證 RAG 檢索結果
    assert isinstance(result_state.get("retrieved_docs", []), list)

    # 驗證資料流動：行程生成工具有用到搜尋結果
    itinerary = result_state["tool_results"]["itinerary"]["result"]
    search_titles = [item["title"] for item in result_state["tool_results"]["search_results"]["result"]]
    for title in search_titles:
        assert title in itinerary or len(search_titles) == 0  # 若有景點，行程應包含景點名

    # 驗證 conversation_summary 欄位存在（如有摘要功能）
    assert "conversation_summary" in result_state
