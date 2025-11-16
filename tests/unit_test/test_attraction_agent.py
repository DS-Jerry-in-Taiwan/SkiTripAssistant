import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import pytest
from graph import attraction_agent
from nodes import search_attractions_node, search_attractions_tool_wrapper
import json

def test_search_attractions_node_basic():
    """
    測試景點搜尋工具基本功能
    """
    result = search_attractions_node("台中滑雪場", top_k=3)
    
    assert "query" in result
    assert "source" in result
    assert "results" in result
    assert isinstance(result["results"], list)
    assert result["query"] == "台中滑雪場"
    
    # 檢查結果格式
    if len(result["results"]) > 0:
        first_result = result["results"][0]
        assert "name" in first_result
        assert "location" in first_result

def test_search_attractions_tool_wrapper():
    """
    測試工具 wrapper 回傳 JSON 字串
    """
    result_json = search_attractions_tool_wrapper("溫泉推薦")
    result = json.loads(result_json)
    
    assert "query" in result
    assert "results" in result
    assert result["query"] == "溫泉推薦"

def test_search_attractions_api_success():
    """
    測試 API 調用成功情境（若 API Key 已設定）
    """
    result = search_attractions_node("Tokyo ski resort", top_k=5)
    
    assert "query" in result
    assert "source" in result
    
    # 若 API 成功，source 應為 google_places_api
    if result["source"] == "google_places_api":
        assert len(result["results"]) > 0
        assert "summary" in result

def test_search_attractions_fallback():
    """
    測試 fallback 機制（API 失敗時使用本地資料）
    """
    # 暫時移除環境變數模擬 API 失敗
    original_key = os.environ.get("GOOGLE_PLACES_API_KEY")
    if original_key:
        del os.environ["GOOGLE_PLACES_API_KEY"]
    
    result = search_attractions_node("滑雪場", top_k=3)
    
    # 檢查是否使用 fallback
    assert "source" in result
    assert result["source"] in ["kb_fallback", "error"]
    
    # 恢復環境變數
    if original_key:
        os.environ["GOOGLE_PLACES_API_KEY"] = original_key

def test_attraction_agent_integration():
    """
    測試 Attraction Agent 整合（含 LLM 推理與工具調用）
    """
    result = attraction_agent.invoke({
        "messages": [{"role": "user", "content": "請查詢台中滑雪場與溫泉推薦"}]
    })
    
    assert "messages" in result
    assert len(result["messages"]) > 0
    
    # 檢查回覆是否包含結構化資訊
    final_response = result["messages"][-1].content
    assert "景點" in final_response or "滑雪" in final_response or "溫泉" in final_response

def test_attraction_agent_json_format():
    """
    檢查 Agent 回傳內容是否包含標準 JSON 格式
    """
    result = attraction_agent.invoke({
        "messages": [{"role": "user", "content": "查詢台中景點推薦"}]
    })
    
    final_content = result["messages"][-1].content
    
    # 嘗試解析 JSON
    try:
        # 移除可能的 markdown 格式
        json_str = final_content.replace("```json", "").replace("```", "").strip()
        json_obj = json.loads(json_str)
        
        # 檢查必要欄位
        assert "查詢任務" in json_obj or "景點清單" in json_obj
    except Exception:
        # 若解析失敗，至少要有關鍵字
        assert "查詢" in final_content or "景點" in final_content

def test_attraction_agent_empty_query():
    """
    測試空查詢處理
    """
    result = attraction_agent.invoke({
        "messages": [{"role": "user", "content": ""}]
    })
    
    assert "messages" in result
    assert len(result["messages"]) > 0

def test_attraction_agent_multiple_keywords():
    """
    測試多關鍵字查詢
    """
    result = attraction_agent.invoke({
        "messages": [{"role": "user", "content": "查詢台中滑雪場、溫泉、美食推薦"}]
    })
    
    final_content = result["messages"][-1].content
    
    # 檢查是否涵蓋多個關鍵字
    assert "滑雪" in final_content or "溫泉" in final_content or "美食" in final_content

def test_attraction_agent_error_handling():
    """
    測試錯誤處理機制
    """
    # 測試異常查詢
    result = search_attractions_node("!@#$%^&*()", top_k=3)
    
    assert "query" in result
    assert "source" in result
    # 應該要有錯誤處理，不會直接崩潰