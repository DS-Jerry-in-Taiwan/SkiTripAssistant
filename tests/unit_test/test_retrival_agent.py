import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import pytest
from graph import retriever_agent

def test_retriever_agent_basic():
    """
    測試 Retriever Agent 是否能正確回傳結構化檢索結果
    """
    query = "查詢滑雪場資訊"
    result = retriever_agent.invoke({"messages": [{"role": "user", "content": query}]})
    assert "messages" in result
    assert len(result["messages"]) > 0
    final_content = result["messages"][-1].content
    assert "檢索" in final_content or "滑雪" in final_content
    assert "JSON" in final_content or "{" in final_content  # 檢查是否有結構化格式

def test_retriever_agent_json_format():
    """
    檢查回傳內容是否包含標準 JSON 格式欄位
    """
    query = "查詢溫泉課程"
    result = retriever_agent.invoke({"messages": [{"role": "user", "content": query}]})
    final_content = result["messages"][-1].content
    # 嘗試解析 JSON
    import json
    try:
        json_obj = json.loads(final_content.replace("```json", "").replace("```", "").strip())
        assert "檢索任務" in json_obj
        assert "檢索結果" in json_obj
    except Exception:
        # 若解析失敗，至少要有關鍵字
        assert "檢索任務" in final_content
        assert "檢索結果" in final_content