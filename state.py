from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict

class AnyMessage(TypedDict):
    role: str
    content: str
    
class Document(TypedDict):
    title: str
    content: str
    source: Optional[str]
    
class ToolResult(TypedDict):
    tool_name: str
    result: Any
    
class AgentState(TypedDict, total=False):
    """
    統一狀態結構，支援所有 agent node、工具、函數的輸入與輸出
    """
    # 對話與查詢
    user_input: str  # 使用者最新輸入
    messages: List[AnyMessage]  # 多輪對話訊息
    query: Optional[str]  # 查詢字串（可累積所有 user query）
    recent_queries: List[str]  # 近期查詢
    conversation_summary: Optional[str]  # 對話摘要

    # 偏好與檢索
    user_preferences: Dict[str, Any]  # 解析後的偏好
    retrieved_docs: List[Document]  # RAG 檢索結果
    retriever_data: Optional[Any]  # 檢索資料（結構化結果）

    # 工具/agent結果
    tool_results: Optional[List[ToolResult]]  # 工具回傳結果（如 search, weather, etc）
    attraction_data: Optional[Any]  # 景點資料
    weather_data: Optional[Any]  # 天氣資料
    accommodation_data: Optional[Any]  # 住宿資料
    planner_result: Optional[Any]  # 行程規劃結果
    evaluation_result: Optional[Any]  # 行程評估結果
    final_itinerary: Optional[Any]  # 最終行程（可用於 output/存檔）

    # 流程控制
    need_planning: Optional[bool]  # 是否進入規劃階段
    need_evaluation: Optional[bool]  # 是否進入評估階段
    current_agent: Optional[str]  # 目前執行 agent
    top_k: Optional[int]  # 檢索/推薦數量（如有需要可用）

    # 其他可擴充欄位
    error: Optional[str]  # 錯誤訊息（如有）
    status: Optional[str]  # 狀態標記（如有）