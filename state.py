from typing import List, Dict, Optional, Any
from typing_extensions import TypedDict, Annotated

class AnyMessage(TypedDict):
    """Single message of any role"""
    role: str
    content: str
    
def add_message(messages: List[AnyMessage], new_message: AnyMessage) -> List[AnyMessage]:
    """Reducer to add a message to the conversation history"""
    return messages + [new_message]

class Document(TypedDict):
    """RAG retrieved document"""
    title: str
    content: str
    source: Optional[str]
    
class ToolResult(TypedDict):
    """Tools result structure"""
    tool_name: str
    result: str

class AgentState(TypedDict):
    """
    State structure for the travel agent
    args:
    messages: List of messages in the conversation
    example: [{"role": "user", "content": "我是一個滑雪新手，我想明年初排去滑雪，有推薦日本的雪場嗎？"},
              {"role": "assistant", "content": "請問您要玩幾天"}]
    user_preferences: User's travel preferences
    example: {"budget": "中等", "travel_dates": "明年一月", "activity_level": "初學者"}
    retrieved_docs: Documents retrieved for RAG
    example: [{"title": "日本滑雪場推薦", "content": "日本有許多優質的滑雪場，如北海道的二世古、長野的白馬等，適合初學者。", "source": "travel_blog_123"},
              {"title": "滑雪裝備指南", "content": "初學者建議租借滑雪裝備，避免購買過多不必要的用品。", "source": "ski_gear_guide"}]
    query: Current user query
    example: "推薦適合初學者的日本滑雪場"
    conversation_summary: Summary of the conversation so far
    example: "用戶是一個滑雪新手，尋找適合初學者的日本滑雪場推薦。"
    """
    messages: List[AnyMessage]
    user_preferences: Dict[str, Any]
    retrieved_docs: List[Document]
    query: Optional[str]
    conversation_summary: Optional[str]  # 新增欄位
    tool_results: Optional[List[ToolResult]]
    
if __name__ == "__main__":
    state: AgentState = {
        "messages": [
            {"role": "user", "content": "我是一個滑雪新手，我想明年初排去滑雪，有推薦日本的雪場嗎？"},
            {"role": "assistant", "content": "請問您要玩幾天"}
            ],
        "user_preferences": {"budget": "中等", "travel_dates": "明年一月", "activity_level": "初學者"},
        "retrieved_docs": [
            {"title": "日本滑雪場推薦", "content": "日本有許多優質的滑雪場，如北海道的二世古、長野的白馬等，適合初學者。", "source": "travel_blog_123"},
            {"title": "滑雪裝備指南", "content": "初學者建議租借滑雪裝備，避免購買過多不必要的用品。", "source": "ski_gear_guide"}
        ],
        "query": "推薦適合初學者的日本滑雪場",
        "conversation_summary": "用戶是一個滑雪新手，尋找適合初學者的日本滑雪場推薦。",
        "tool_results": {
            "search_results": {
                "tool_name": "tavily_search",
                "result": [
                    {"title": "上野公園", "description": "東京最著名的賞櫻地點之一。", "url": "https://example.com/ueno", "rating": 4.8},
                    {"title": "目黑川", "description": "沿岸櫻花盛開，適合散步。", "url": "https://example.com/meguro", "rating": 4.7}
                ]
            }
        }
    }
    print("AgentState example:")
    for key, value in state.items():
        print(f"{key}: {value}")
    
