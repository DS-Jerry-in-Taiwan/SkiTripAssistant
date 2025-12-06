import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import pytest
import signal
from graph import itinerary_agent
from nodes import (
    format_itinerary_data,
    calculate_budget_estimate,
    calculate_route,
    get_weather_forecast,
    search_accommodation
)
import json

def timeout_handler(signum, frame):
    raise TimeoutError("測試超時")

# ========== 工具函數測試 ==========

# def test_format_itinerary_data():
#     """測試資料格式化工具"""
#     retriever_data = json.dumps({
#         "results": [
#             {"title": "滑雪場介紹", "content": "雪山滑雪場"}
#         ]
#     })
    
#     attraction_data = json.dumps({
#         "results": [
#             {"name": "雪山滑雪場", "location": "台中"}
#         ]
#     })
    
#     result_json = format_itinerary_data(retriever_data, attraction_data)
#     result = json.loads(result_json)
    
#     assert "知識庫資訊" in result
#     assert "景點資訊" in result
#     assert "摘要" in result


# def test_calculate_budget_estimate():
#     """測試預算估算工具"""
#     result_json = calculate_budget_estimate(3, "中等")
#     result = json.loads(result_json)
    
#     assert result["天數"] == 3
#     assert result["預算等級"] == "中等"
#     assert "總預算估算" in result
#     assert "預算明細" in result


# def test_calculate_route():
#     """測試路線規劃工具"""
#     result_json = calculate_route("台北車站", "台中車站", "transit")
#     result = json.loads(result_json)
    
#     # 若 API Key 存在且有效
#     if "error" not in result:
#         assert "起點" in result
#         assert "終點" in result
#         assert "距離" in result
#         assert "預估時間" in result


# def test_get_weather_forecast():
#     """測試天氣查詢工具"""
#     from datetime import datetime, timedelta
#     tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
#     result_json = get_weather_forecast("台中", tomorrow)
#     result = json.loads(result_json)
    
#     # 若 API Key 存在且有效
#     if "error" not in result:
#         assert "地點" in result
#         assert "日期" in result
#         assert "預報" in result


# def test_search_accommodation():
#     """測試住宿查詢工具"""
#     result_json = search_accommodation("台中", "2024-12-20", "2024-12-22")
#     result = json.loads(result_json)
    
#     assert "地點" in result
#     assert "入住日期" in result
#     assert "推薦住宿" in result
#     assert isinstance(result["推薦住宿"], list)


# ========== Itinerary Agent 整合測試 ==========

# def test_itinerary_agent_basic():
#     """測試 Itinerary Agent 基本功能"""
#     request = """
#     我想去台中玩，請幫我安排一個行程！
#     想去台中溫泉會館，
#     預計玩2天1夜，預算中等，
#     出發日期是2024-12-20，請幫我詳細規劃一下。
#     """
    
#     result = itinerary_agent.invoke({
#         "messages": [{"role": "user", "content": request}]
#     })
    
#     assert "messages" in result
#     assert len(result["messages"]) > 0
    
#     final_content = result["messages"][-1].content
#     assert "行程" in final_content or "Day" in final_content


def test_itinerary_agent_with_tools():
    """測試 Itinerary Agent 是否正確調用工具"""
    request = """
        請規劃台中3天2夜行程，預算中等。
        請使用 calculate_budget 估算預算。
    """
        # 設定 30 秒 timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        result = itinerary_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        },
            config={"recursion_limit": 15}
        )
        
        # 檢查是否有 tool calls
        messages = result["messages"]
        has_tool_calls = any(
            hasattr(msg, 'tool_calls') and msg.tool_calls 
            for msg in messages
        )
        
        # 若 Agent 正確使用工具，應該有 tool_calls
        assert has_tool_calls or "預算" in result["messages"][-1].content
    
    finally:
        signal.alarm(0)  # 取消 timeout


# def test_itinerary_agent_json_format():
#     """測試回覆格式是否符合 JSON 規範"""
#     request = "生成台中3天2夜滑雪行程，預算中等"
    
#     result = itinerary_agent.invoke({
#         "messages": [{"role": "user", "content": request}]
#     })
    
#     final_content = result["messages"][-1].content
    
#     # 嘗試解析 JSON
#     try:
#         json_str = final_content.replace("```json", "").replace("```", "").strip()
#         json_obj = json.loads(json_str)
#         assert "行程規劃" in json_obj or "itinerary" in json_obj
#     except:
#         # 若解析失敗，至少要有關鍵字
#         assert "行程" in final_content or "Day" in final_content


# def test_itinerary_agent_with_data():
#     """測試 Itinerary Agent 整合 Retriever 與 Attraction 資料"""
#     retriever_data = json.dumps({
#         "results": [
#             {"title": "滑雪場介紹", "content": "雪山滑雪場提供初學者課程"}
#         ]
#     })
    
#     attraction_data = json.dumps({
#         "results": [
#             {"name": "雪山滑雪場", "location": "台中", "rating": 4.5}
#         ]
#     })
    
#     request = f"""
# 請根據以下資料生成行程：

# 檢索資訊：{retriever_data}
# 景點資訊：{attraction_data}

# 請規劃2天1夜行程，預算中等。
# """
    
#     result = itinerary_agent.invoke({
#         "messages": [{"role": "user", "content": request}]
#     })
    
#     final_content = result["messages"][-1].content
#     assert "雪山滑雪場" in final_content or "滑雪" in final_content


# # ========== 錯誤處理測試 ==========

# def test_itinerary_agent_empty_input():
#     """測試空輸入處理"""
#     result = itinerary_agent.invoke({
#         "messages": [{"role": "user", "content": ""}]
#     })
    
#     assert "messages" in result
#     assert len(result["messages"]) > 0


# def test_itinerary_agent_invalid_data():
#     """測試無效資料處理"""
#     request = """
# 請根據以下資料生成行程：
# 檢索資訊：{invalid json}
# 景點資訊：{invalid json}
# """
    
#     result = itinerary_agent.invoke({
#         "messages": [{"role": "user", "content": request}]
#     })
    
#     # 應該要能處理無效資料，不會崩潰
#     assert "messages" in result