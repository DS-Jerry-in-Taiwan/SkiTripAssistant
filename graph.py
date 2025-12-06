import os
import json
from nodes import update_recent_queries, get_merged_query, dynamic_k_by_query
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from typing import List, Optional, Dict, Any
from nodes import (
    search_attractions_tool_wrapper,
    rag_retrieval_tool_wrapper,
    format_itinerary_data_wrapper,
    calculate_budget_wrapper,
    calculate_route_wrapper,
    get_weather_wrapper,
    search_accommodation_wrapper,
    parse_user_preferences_wrapper
)
from travel_agent_mvp.tools import (
    retriever_base_tools,
    attraction_base_tools,
    itinerary_base_tools,
    weather_base_tools,
    accommodation_base_tools
)
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ========== Initialize LLMs ==========
planner_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.7, openai_api_key=OPENAI_API_KEY)
retriever_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3, openai_api_key=OPENAI_API_KEY)
attraction_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3, openai_api_key=OPENAI_API_KEY)
itinerary_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.7, openai_api_key=OPENAI_API_KEY)
evaluator_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.5, openai_api_key=OPENAI_API_KEY)
weather_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3, openai_api_key=OPENAI_API_KEY)
accommodation_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3, openai_api_key=OPENAI_API_KEY)
evaluator_llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3, openai_api_key=OPENAI_API_KEY)
recommendation_llm = ChatOpenAI(model=OPENAI_MODEL,temperature=0.5,openai_api_key=OPENAI_API_KEY)

def load_prompt_template(filename: str) -> str:
    """ load prompt template from file """
    template_path = Path(__file__).parent() / "prompts" / filename
    with open(template_path, "r", encoding="utf-8") as file:
        return file.read()

# ========== Create Base Agents First ==========

retriever_agent = create_agent(
    model=retriever_llm,
    tools=retriever_base_tools,
    system_prompt=load_prompt_template("retriever.txt")
)

attraction_agent = create_agent(
    model=attraction_llm,
    tools=attraction_base_tools,
    system_prompt=load_prompt_template("attraction.txt")
)

itinerary_agent = create_agent(
    model=itinerary_llm,
    tools=itinerary_base_tools,
    system_prompt=load_prompt_template("itinerary.txt"),
    response_format=ItineraryOutput,
    debug=False
)

weather_agent = create_agent(
    model=weather_llm,
    tools=weather_base_tools,
    system_prompt=load_prompt_template("weather.txt"),
    response_format=WeatherOutput,
    debug=False
)

accommodation_agent = create_agent(
    model=accommodation_llm,
    tools=accommodation_base_tools,
    system_prompt=load_prompt_template("accommodation.txt"),
    response_format=AccommodationOutput,
    debug=False
)

evaluator_agent = create_agent(
    model=evaluator_llm,
    # tools=evaluator_tools,
    system_prompt=load_prompt_template("evaluator.txt"),
    response_format=EvaluatorOutput,
    debug=False
)



# ========== Agent Communication Tools (A2A) ==========

def call_retriever_agent(query: str) -> str:
    """呼叫完整的 Retriever Agent（包含推理循環）"""
    query = get_merged_query(state)
    result = retriever_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

def call_attraction_agent(query: str) -> str:
    """呼叫完整的 Attraction Agent（包含推理循環）"""
    result = attraction_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

def call_itinerary_agent(user_request: str, retriever_data: str = "", attraction_data: str = "") -> str:
    f"""
    使用者需求：{user_request}
    檢索資訊（JSON）：{retriever_data}
    景點資訊（JSON）：{attraction_data}

    請依照以下格式回覆，所有欄位都需填寫：
    {{
      "summary": "...",
      "total_budget": "...",
      "transport_plan": "...",
      "days": ...,
      "budget_level": "...",
      "daily_plans": [
        {{
          "date": "...",
          "activities": [
            {{
              "time": "...",
              "activity": "...",
              "location": "...",
              "transport": "...",
              "notes": "..."
            }}
          ],
          "meals": {{
            "breakfast": "...",
            "lunch": "...",
            "dinner": "..."
          }},
          "accommodation": "..."
        }}
      ]
    }}
    請直接回覆符合格式的 JSON，不要加自然語言說明。
    """
    result = itinerary_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return result["messages"][-1].content

def call_weather_agent(location: str, start_date: str, end_date: str = None) -> str:
    """
    呼叫 Weather Agent 查詢天氣資訊
    
    Args:
        location: 地點（如「台中」、「台北」）
        start_date: 開始日期（格式 YYYY-MM-DD）
        end_date: 結束日期（可選，格式 YYYY-MM-DD）
    
    Returns:
        JSON 格式的天氣預報資訊
    """
    if end_date:
        query = f"請查詢 {location} 從 {start_date} 到 {end_date} 的天氣預報"
    else:
        query = f"請查詢 {location} 在 {start_date} 的天氣預報"
    
    result = weather_agent.invoke({"messages": [{"role": "user", "content": query}]})
    # 找到最後一個 AIMessage
    for msg in reversed(result["messages"]):
        if hasattr(msg, 'content'):
            content = msg.content
            
            # 如果是 Pydantic 物件，直接轉 JSON
            if hasattr(content, 'model_dump'):
                weather_data = content.model_dump()
                return json.dumps(weather_data, ensure_ascii=False, indent=2)
            
            # 如果是字串，嘗試解析
            # 處理字串格式
            if isinstance(content, str):
                # 移除 debug 輸出前綴
                if "Returning structured response:" in content:
                    # 嘗試從字串中提取結構化資料
                    # 格式: "Returning structured response: 查詢地點='台中' 天氣預報=[...] 整體分析='...'"
                    try:
                        # 使用 eval 解析 (僅用於已知格式)
                        import re
                        # 提取查詢地點
                        location_match = re.search(r"查詢地點='([^']+)'", content)
                        # 提取整體分析
                        analysis_match = re.search(r"整體分析='([^']+)'", content)
                        
                        # 提取天氣預報列表
                        forecast_match = re.search(r"天氣預報=\[(.*?)\] 整體分析", content, re.DOTALL)
                        
                        if location_match and analysis_match and forecast_match:
                            forecast_str = forecast_match.group(1)
                            # 解析每個 WeatherForecast 物件
                            forecasts = []
                            for item in re.finditer(r"WeatherForecast\(日期='([^']+)', 天氣='([^']+)', 氣溫='([^']+)'\)", forecast_str):
                                forecasts.append({
                                    "日期": item.group(1),
                                    "天氣": item.group(2),
                                    "氣溫": item.group(3)
                                })
                            
                            weather_data = {
                                "查詢地點": location_match.group(1),
                                "天氣預報": forecasts,
                                "整體分析": analysis_match.group(1)
                            }
                            return json.dumps(weather_data, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"警告：無法解析 structured response: {e}")
                
                # 嘗試直接解析為 JSON
                try:
                    # 如果已經是 JSON 格式
                    data = json.loads(content)
                    return json.dumps(data, ensure_ascii=False, indent=2)
                except:
                    pass
                
                # 清理 markdown
                if "```json" in content:
                    content = content.split("```json")[-1]
                if "```" in content:
                    content = content.split("```")[0]
                content = content.strip()
                
                # 只取 JSON 部分
                if "{" in content and "}" in content:
                    content = content[content.find("{"):content.rfind("}")+1]
                
                return content
    
    return json.dumps({"error": "無法解析天氣資料"}, ensure_ascii=False, indent=2)

def call_accommodation_agent(location: str, checkin: str, checkout: str) -> str:
    """
    呼叫 Accommodation Agent 查詢住宿資訊
    
    Args:
        location: 地點（如「台中」、「台北」）
        checkin: 入住日期（格式 YYYY-MM-DD）
        checkout: 退房日期（格式 YYYY-MM-DD）
    
    Returns:
        JSON 格式的住宿推薦資訊
    """
    if not checkin or not checkout:
        raise ValueError("請提供入住與退房日期")
        return "請提供入住與退房日期（checkin/checkout）"
    query = f"請查詢 {location} 的住宿,入住日期 {checkin}，退房日期 {checkout}"
    
    result = accommodation_agent.invoke({"messages": [{"role": "user", "content": query}]})
    
    # 找到最後一個 AIMessage
    for msg in reversed(result["messages"]):
        if hasattr(msg, 'content'):
            content = msg.content
            
            # 處理 Pydantic 物件 (AccommodationOutput)
            if hasattr(content, 'model_dump'):
                accommodation_data = content.model_dump()
                return json.dumps(accommodation_data, ensure_ascii=False, indent=2)
            
            # 處理字串格式
            if isinstance(content, str):
                # 移除 debug 輸出前綴並解析結構化資料
                if "Returning structured response:" in content:
                    try:
                        import re
                        
                        # 提取查詢地點
                        location_match = re.search(r"查詢地點='([^']+)'", content)
                        # 提取入住日期
                        checkin_match = re.search(r"入住日期='([^']+)'", content)
                        # 提取退房日期
                        checkout_match = re.search(r"退房日期='([^']+)'", content)
                        # 提取整體分析
                        analysis_match = re.search(r"整體分析='([^']+)'", content)
                        
                        # 提取推薦住宿列表
                        hotels = []
                        hotel_pattern = r"HotelRecommendation\(名稱='([^']+)', 類型='([^']+)', 評分=([\d.]+), 價格='([^']+)', 總價='([^']+)', 特色=\[([^\]]+)\], 交通='([^']+)', 推薦理由='([^']+)'\)"
                        for hotel_match in re.finditer(hotel_pattern, content):
                            # 解析特色列表
                            features_str = hotel_match.group(6)
                            features = [f.strip().strip("'\"") for f in features_str.split(',')]
                            
                            hotels.append({
                                "名稱": hotel_match.group(1),
                                "類型": hotel_match.group(2),
                                "評分": float(hotel_match.group(3)),
                                "價格": hotel_match.group(4),
                                "總價": hotel_match.group(5),
                                "特色": features,
                                "交通": hotel_match.group(7),
                                "推薦理由": hotel_match.group(8)
                            })
                        
                        # 提取選擇建議
                        suggestion_match = re.search(
                            r"選擇建議=ChoiceSuggestion\(預算型='([^']+)', 體驗型='([^']+)', 家庭型='([^']+)'\)",
                            content
                        )
                        
                        if location_match and checkin_match and checkout_match and analysis_match:
                            accommodation_data = {
                                "查詢地點": location_match.group(1),
                                "入住日期": checkin_match.group(1),
                                "退房日期": checkout_match.group(1),
                                "推薦住宿": hotels,
                                "選擇建議": {
                                    "預算型": suggestion_match.group(1) if suggestion_match else "",
                                    "體驗型": suggestion_match.group(2) if suggestion_match else "",
                                    "家庭型": suggestion_match.group(3) if suggestion_match else ""
                                },
                                "整體分析": analysis_match.group(1)
                            }
                            return json.dumps(accommodation_data, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"警告：無法解析 structured response: {e}")
                
                # 嘗試直接解析為 JSON
                try:
                    data = json.loads(content)
                    return json.dumps(data, ensure_ascii=False, indent=2)
                except:
                    pass
                
                # 清理 markdown
                if "```json" in content:
                    content = content.split("```json")[-1]
                if "```" in content:
                    content = content.split("```")[0]
                content = content.strip()
                
                # 只取 JSON 部分
                if "{" in content and "}" in content:
                    content = content[content.find("{"):content.rfind("}")+1]
                    try:
                        data = json.loads(content)
                        return json.dumps(data, ensure_ascii=False, indent=2)
                    except:
                        pass
                
                return content
    
    return json.dumps({
        "error": "無法解析住宿資料",
        "查詢地點": location,
        "入住日期": checkin,
        "退房日期": checkout,
        "推薦住宿": [],
        "選擇建議": {
            "預算型": "查詢失敗",
            "體驗型": "查詢失敗",
            "家庭型": "查詢失敗"
        },
        "整體分析": "查詢失敗"
    }, ensure_ascii=False, indent=2)
    
def call_evaluator_agent(itinerary_data: dict, user_preferences: dict) -> str:
    """
    呼叫 Evaluator Agent 評估行程
    Returns: JSON 格式的評估結果
    """
    query = f"""
    請評估以下行程：
    ## 行程資料
    {json.dumps(itinerary_data, ensure_ascii=False, indent=2)}
    ## 使用者偏好
    {json.dumps(user_preferences, ensure_ascii=False, indent=2)}
    請根據上述資料，評估行程品質並提供優化建議。
    """
    result = evaluator_agent.invoke({"messages": [{"role": "user", "content": query}]})

    # 找到最後一個 AIMessage
    for msg in reversed(result["messages"]):
        if hasattr(msg, 'content'):
            content = msg.content

            # 如果是 Pydantic 物件，直接轉 JSON
            if hasattr(content, 'model_dump_json'):
                return content.model_dump_json(indent=2, ensure_ascii=False)
            if hasattr(content, 'model_dump'):
                return json.dumps(content.model_dump(), ensure_ascii=False, indent=2)

            # 如果是字串，嘗試解析
            if isinstance(content, str):
                # 處理 "Returning structured response:" 格式
                if "Returning structured response:" in content:
                    import re
                    # 提取各欄位
                    summary = re.search(r"行程摘要='([^']+)'", content)
                    score = re.search(r"評分=ItineraryScore\((.*?)\)", content)
                    suggestions = re.findall(r"OptimizationSuggestion\((.*?)\)", content)
                    evaluation = re.search(r"整體評價='([^']+)'", content)
                    need_adjust = re.search(r"是否需要調整=(True|False)", content)

                    # 解析評分
                    score_dict = {}
                    if score:
                        for item in score.group(1).split(','):
                            k, v = item.split('=')
                            score_dict[k.strip()] = float(v.strip())
                    # 解析優化建議
                    suggestion_list = []
                    for s in suggestions:
                        fields = re.findall(r"(\w+)='([^']+)'", s)
                        suggestion_list.append({k: v for k, v in fields})

                    # 組合 JSON
                    result_json = {
                        "行程摘要": summary.group(1) if summary else "",
                        "評分": score_dict,
                        "優化建議": suggestion_list,
                        "整體評價": evaluation.group(1) if evaluation else "",
                        "是否需要調整": True if need_adjust and need_adjust.group(1) == "True" else False
                    }
                    return json.dumps(result_json, ensure_ascii=False, indent=2)

                # 嘗試直接解析為 JSON
                try:
                    data = json.loads(content)
                    return json.dumps(data, ensure_ascii=False, indent=2)
                except:
                    pass

                # 清理 markdown
                if "```json" in content:
                    content = content.split("```json")[-1]
                if "```" in content:
                    content = content.split("```")[0]
                content = content.strip()

                # 只取 JSON 部分
                if "{" in content and "}" in content:
                    content = content[content.find("{"):content.rfind("}")+1]
                    try:
                        data = json.loads(content)
                        return json.dumps(data, ensure_ascii=False, indent=2)
                    except:
                        pass

                return content

    return json.dumps({
        "error": "無法解析評估結果",
        "行程摘要": "評估失敗",
        "評分": {
            "預算合理性": 0,
            "時間安排": 0,
            "交通便利性": 0,
            "活動豐富度": 0,
            "整體評分": 0
        },
        "優化建議": [],
        "整體評價": "評估失敗",
        "是否需要調整": False
    }, ensure_ascii=False, indent=2)

# ========== Define A2A Tools ==========

planner_tools = [
    Tool(
        name="call_retriever",
        func=call_retriever_agent,
        description="呼叫 Retriever Agent 檢索知識庫資訊。輸入：查詢問題（字串）"
    ),
    Tool(
        name="call_attraction",
        func=call_attraction_agent,
        description="呼叫 Attraction Agent 查詢景點資訊。輸入：查詢問題（字串）"
    ),
    # Tool(
    #     name="call_itinerary",
    #     func=call_itinerary_agent,
    #     description="呼叫 Itinerary Agent 生成行程規劃。輸入：user_request, retriever_data, attraction_data"
    # ),
    Tool(
        name="call_weather",
        func=call_weather_agent,
        description="""
        呼叫 Weather Agent 查詢天氣預報。
        輸入：location（地點）, start_date（開始日期 YYYY-MM-DD）, end_date（結束日期，可選）
        輸出：天氣預報與活動建議（JSON）
        """
    ),
    Tool(
        name="call_accommodation",
        func=call_accommodation_agent,
        description="""
        呼叫 Accommodation Agent 查詢住宿推薦。
        輸入：location（地點）, checkin（入住日期 YYYY-MM-DD）, checkout（退房日期 YYYY-MM-DD）
        輸出：住宿清單與推薦建議（JSON）
        """
    )
]

evaluator_tools = []

recommendation_tools = [
    Tool(
        name="call_retriever",
        func=call_retriever_agent,
        description="呼叫 Retriever Agent 檢索知識庫資訊。輸入：查詢問題（字串）"
    ),
    Tool(
        name="call_attraction",
        func=call_attraction_agent,
        description="呼叫 Attraction Agent 查詢景點資訊。輸入：查詢問題（字串）"
    ),
    # Tool(
    #     name="call_itinerary",
    #     func=call_itinerary_agent,
    #     description="呼叫 Itinerary Agent 生成行程規劃。輸入：user_request, retriever_data, attraction_data"
    # ),
    StructuredTool.from_function(
        func=call_weather_agent,
        name="call_weather",
        description="""
        呼叫 Weather Agent 查詢天氣預報。
        必須提供：location（地點）, start_date（開始日期 YYYY-MM-DD）, end_date（結束日期，可選）。
        輸出：天氣預報與活動建議（JSON）
        """,
        args_schema=CallWeatherInput
    ),
    StructuredTool.from_function(
        func=call_accommodation_agent,
        name="call_accommodation",
        description="""
        呼叫 Accommodation Agent 查詢住宿推薦。
        必須提供：location（地點）, checkin（入住日期 YYYY-MM-DD）, checkout（退房日期 YYYY-MM-DD）。
        輸出：住宿清單與推薦建議（JSON）
        """,
        args_schema=CallAccommodationInput
    )
]


# ========== Create Coordinator Agents with A2A Tools ==========

planner_agent = create_agent(
    model=planner_llm,
    tools=planner_tools,
    system_prompt=load_prompt_template("planner.txt")
)

recommendation_agent = create_agent(
    model=recommendation_llm,
    tools=recommendation_tools,
    system_prompt=load_prompt_template("recommendation.txt"),
    debug=False
)

# ========== Agent Wrapper Nodes ==========

def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner Agent Node - 協調所有 Agent 並生成行程
    
    工作流程：
    1. 理解使用者需求
    2. 呼叫 Retriever、Attraction、Weather、Accommodation Agent
    3. 呼叫 Itinerary Agent 生成行程
    4. 回傳最終行程
    """
    # update state with user input
    update_recent_queries(state, user_input, max_n=3)
    
    user_input = state.get("user_input", "")
    
    # Planner Agent 會透過 tool calling 自主呼叫其他 Agent
    result = planner_agent.invoke({"messages": [{"role": "user", "content": user_input}]})
    
    # 取得最終回應（可能經過多輪 tool calling）
    final_message = result["messages"][-1]
    
    state["planner_result"] = final_message.content if hasattr(final_message, 'content') else str(final_message)
    state["current_agent"] = "planner"
    state["conversation_history"] = result["messages"]
    
    return state

def evaluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluator Agent Node - 評估行程品質（只評估一次，不觸發重新規劃）
    
    工作流程：
    1. 接收 Planner 生成的行程
    2. 評估行程品質（預算、時間、交通、活動）
    3. 提供優化建議
    4. 回傳評估結果（不觸發重新規劃）
    """
    user_input = state.get("user_input", "")
    planner_result = state.get("planner_result", "")
    
    # 構建評估 prompt
    prompt = f"""
    使用者需求：{user_input}
    
    Planner 生成的行程：
    {planner_result}
    
    請評估行程品質，並提供優化建議。
    注意：本次只評估，不觸發重新規劃。
    """
    
    result = evaluator_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    
    # 取得評估結果
    last_message = result["messages"][-1]
    final_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # 更新 state
    state["evaluation_result"] = final_content
    state["current_agent"] = "evaluator"
    state["final_itinerary"] = planner_result  # 保留原始行程
    
    # 不再設置 need_optimization，直接結束
    state["need_optimization"] = False
    
    return state

def recommendation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    update_recent_queries(state, user_input, max_n=3)
    
    user_input = state.get("user_input", "")
    all_messages = state.get("messages", [])
    context_str = " ".join([msg["content"] for msg in all_messages if msg.get("role") == "user"])
    state["need_planning"] = False

    # 1. 呼叫 recommendation_agent，傳入完整 messages（多輪上下文）
    result = recommendation_agent.invoke({"messages": all_messages})

    final_message = result["messages"][-1]
    rec_reply = final_message.content if hasattr(final_message, 'content') else str(final_message)

    # 2. 把結果丟給 planner_agent，確認資訊是否足夠
    planner_check = planner_agent.invoke({"messages": [{"role": "user", "content": context_str}]})
    planner_reply = planner_check["messages"][-1].content if hasattr(planner_check["messages"][-1], 'content') else str(planner_check["messages"][-1])

    import json
    try:
        if "```json" in planner_reply:
            planner_reply = planner_reply.split("```json")[-1].split("```")[0].strip()
        planner_json = json.loads(planner_reply)
        if "最終回答" in planner_json and planner_json["最終回答"]:
            state["recommendation_result"] = f"✅ 資訊已完整，以下是為您規劃的行程：\n\n{planner_json['最終回答']}"
            state["need_planning"] = True
            state["current_agent"] = "planner"
            return state
        missing_info = planner_json.get("反思", "")
        if missing_info:
            state["recommendation_result"] = f"{rec_reply}\n\n💡 規劃助理提示：{missing_info}"
        else:
            state["recommendation_result"] = rec_reply
        state["need_planning"] = False
        state["current_agent"] = "recommendation"
        return state
    except Exception as e:
        print(f"JSON 解析失敗: {e}")
        state["recommendation_result"] = rec_reply
        state["need_planning"] = False
        state["current_agent"] = "recommendation"
        return state

# ========== Simplified Router（移除優化循環）==========
def route(state: Dict[str, Any]) -> str:
    """
    多 agent 互動分流 router
    - recommendation: 推薦/引導階段
    - planner: 規劃階段
    - evaluator: 評估階段
    - END: 結束
    """
    current_agent = state.get("current_agent", "")
    need_planning = state.get("need_planning", False)
    need_evaluation = state.get("need_evaluation", False)
    # 初始進入推薦 agent
    if current_agent == "" or current_agent == "recommendation":
        if need_planning:
            print("→ 進入 planner")
            return "planner"
        else:
            return END  # 不要回到 recommendation，直接結束，等待使用者下次輸入
    elif current_agent == "planner":
        if need_evaluation:
            print("→ 進入 evaluator")
            return "evaluator"
        else:
            print("→ 結束")
            return END
    elif current_agent == "evaluator":
        print("→ 結束")
        return END
    
    print("→ 預設結束")
    return END
# ========== Create A2A Multi-Agent Workflow ==========
def create_graph() -> StateGraph:
    """
    完整版 Multi-Agent 工作流

    流程：User Input → Recommendation → Planner → Evaluator → END
    - Recommendation Agent：推薦/引導，收集資訊
    - Planner Agent：規劃行程
    - Evaluator Agent：評估行程品質
    - Router 根據 state 分流
    """
    workflow = StateGraph(dict)

    # 加入 agent nodes
    workflow.add_node("recommendation", recommendation_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("evaluator", evaluator_node)

    # 設定 entry point
    workflow.set_entry_point("recommendation")

    # 分流邏輯
    workflow.add_conditional_edges(
        "recommendation",
        route,
        {
            "planner": "planner",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "planner",
        route,
        {
            "evaluator": "evaluator",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "evaluator",
        route,
        {
            END: END
        }
    )

    # 編譯 workflow，啟用記憶體
    memory_saver = MemorySaver()
    return workflow.compile(checkpointer=memory_saver)

# ========== Initialize Graph ==========
mvp_graph = create_graph()