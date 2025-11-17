import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import pytest
import nodes

def test_tool_integration(monkeypatch):
    # 模擬工具節點回傳
    dummy_search_results = [
        {"title": "東京塔", "description": "地標", "url": "https://tokyotower.jp", "score": 4.8}
    ]
    dummy_llm_response = {"role": "assistant", "content": "推薦您參觀東京塔並安排一日遊行程。"}

    def mock_call_search_tool(state, api_key=None):
        state.setdefault("tool_results", {})
        state["tool_results"]["search_results"] = {
            "tool_name": "tavily_search",
            "result": dummy_search_results
        }
        print("=== MOCK SEARCH NODE CALLED ===")
        return state

    def mock_call_model_node(state):
        state.setdefault("messages", [])
        state["messages"].append(dummy_llm_response)
        print("=== MOCK LLM NODE CALLED ===")
        return state

    # 只 monkeypatch現有工具節點
    monkeypatch.setattr(nodes, "call_search_tool", mock_call_search_tool)
    monkeypatch.setattr(nodes, "call_model_node", mock_call_model_node)

    # 建立 workflow
    from graph import create_graph
    app = create_graph()

    state = {
        "query": "東京景點",
        "user_preferences": {"location": "東京"},
        "tool_results": {},
        "retrieved_docs": [],
        "messages": [],
        "thread_id": "integration-thread-001"
    }

    result = app.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    print("===================", result, "===================")

    # 驗證工具間資料流動與整合
    assert "search_results" in result["tool_results"]
    assert result["tool_results"]["search_results"]["result"] == dummy_search_results
    assert any(msg["role"] == "assistant" for msg in result["messages"])