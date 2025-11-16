import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import pytest
import nodes

def test_graph_workflow(monkeypatch):
    # 模擬工具節點回傳
    dummy_search_results = [
        {"title": "東京塔", "description": "地標", "url": "https://tokyotower.jp", "score": 4.8}
    ]
    dummy_rag_results = [
        {"title": "東京旅遊", "content": "推薦東京塔", "source": "blog"}
    ]
    dummy_llm_response = {"role": "assistant", "content": "推薦您參觀東京塔。"}

    # mock search node
    def mock_call_search_tool(state, api_key=None):
        state.setdefault("tool_results", {})
        state["tool_results"]["search_results"] = {
            "tool_name": "tavily_search",
            "result": dummy_search_results
        }
        print("=== MOCK SEARCH NODE CALLED ===")
        return state

    # mock rag node
    def mock_rag_retrieval_node(state):
        state["retrieved_docs"] = dummy_rag_results
        print("=== MOCK RAG NODE CALLED ===")
        return state

    # mock llm node
    def mock_call_model_node(state):
        state.setdefault("messages", [])
        state["messages"].append(dummy_llm_response)
        print("=== MOCK LLM NODE CALLED ===")
        return state

    # monkeypatch節點
    monkeypatch.setattr(nodes, "call_search_tool", mock_call_search_tool)
    monkeypatch.setattr(nodes, "rag_retrieval_node", mock_rag_retrieval_node)
    monkeypatch.setattr(nodes, "call_model_node", mock_call_model_node)

    from graph import create_graph
    # 建立 workflow
    app = create_graph()

    # 初始化狀態
    state = {
        "query": "東京景點",
        "user_preferences": {"location": "東京"},
        "tool_results": {},
        "retrieved_docs": [],
        "messages": [],
        "thread_id": "test-thread-001"  # 新增這一行
    }

    # 執行 workflow
    result = app.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})

    
    # 驗證各節點資料流動
    assert "search_results" in result["tool_results"]
    assert result["tool_results"]["search_results"]["result"] == dummy_search_results
    assert result["retrieved_docs"] == dummy_rag_results
    assert any(msg["role"] == "assistant" for msg in result["messages"])