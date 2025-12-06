import os
import json
from typing import Dict, Any
from state import AgentState

# ========== State Helpers ==========
def update_recent_queries(state: AgentState, query: str, max_n: int = 3) -> AgentState:
    """更新最近查詢記錄"""
    recent = state.get("recent_queries", [])
    recent.append(query)
    state["recent_queries"] = recent[-max_n:]
    return state

def get_merged_query(state: AgentState) -> str:
    """合併所有查詢字串"""
    queries = state.get("recent_queries", [])
    return " ".join(queries)

def dynamic_k_by_query(state: AgentState) -> int:
    """根據查詢內容動態決定檢索數量"""
    query = state.get("user_input", "")
    if "滑雪" in query:
        return 5
    return 3

# ========== Tool Wrappers ==========
def search_attractions_tool_wrapper(state: AgentState) -> AgentState:
    """景點查詢工具包裝器"""
    query = state.get("user_input", "")
    # TODO: 實作 API 或本地資料查詢
    state["attraction_result"] = {"result": "景點查詢結果"}
    return state

def rag_retrieval_tool_wrapper(state: AgentState) -> AgentState:
    """知識庫檢索工具包裝器"""
    query = state.get("user_input", "")
    # TODO: 實作 Chroma DB 檢索
    state["retriever_result"] = {"result": "檢索結果"}
    return state

def format_itinerary_data_wrapper(state: AgentState) -> AgentState:
    """行程資料格式化包裝器"""
    retriever_data = state.get("retriever_result", {})
    attraction_data = state.get("attraction_result", {})
    # TODO: 整合格式化資料
    state["formatted_data"] = {"result": "格式化後資料"}
    return state

def calculate_budget_wrapper(state: AgentState) -> AgentState:
    """預算計算工具包裝器"""
    days = state.get("days", 1)
    budget_level = state.get("budget_level", "中等")
    # TODO: 實作預算計算
    state["budget_result"] = {"total_budget": 15000, "level": budget_level}
    return state

def calculate_route_wrapper(state: AgentState) -> AgentState:
    """路線計算工具包裝器"""
    origin = state.get("origin", "")
    destination = state.get("destination", "")
    mode = state.get("mode", "transit")
    # TODO: 實作路線查詢
    state["route_result"] = {"route": f"{origin} → {destination}", "mode": mode}
    return state

def get_weather_wrapper(state: AgentState) -> AgentState:
    """天氣查詢工具包裝器"""
    location = state.get("location", "")
    date = state.get("date", "")
    # TODO: 實作天氣查詢
    state["weather_result"] = {"location": location, "date": date, "forecast": "晴天"}
    return state

def search_accommodation_wrapper(state: AgentState) -> AgentState:
    """住宿查詢工具包裝器"""
    location = state.get("location", "")
    checkin = state.get("checkin", "")
    checkout = state.get("checkout", "")
    # TODO: 實作住宿查詢
    state["accommodation_result"] = {"hotels": ["飯店A", "飯店B"]}
    return state

def parse_user_preferences_wrapper(state: AgentState) -> AgentState:
    """使用者偏好解析工具包裝器"""
    raw_text = state.get("user_input", "")
    required_fields = state.get("required_fields", ["days", "budget_level", "date", "location"])
    # TODO: 實作偏好解析
    state["parsed_preferences"] = {"days": 3, "budget_level": "中等", "date": "2025-01-15", "location": "台中"}
    return state

# ========== 其餘函數請註解 ==========
# def unused_function_x(...):
#     # TODO: 待確認用途
#     pass

# def legacy_tool_y(...):
#     # TODO: 已棄用，暫時保留
#     pass

# ...existing code...