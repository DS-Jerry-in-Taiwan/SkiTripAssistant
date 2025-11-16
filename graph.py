import os
import json
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
# ========= Structured Output Schema for Itinerary Agent =========
class Activity(BaseModel):
    time: Optional[str]
    activity: str
    location: str
    transport: Optional[str]
    notes: Optional[str]

class Meals(BaseModel):
    breakfast: Optional[str]
    lunch: Optional[str]
    dinner: Optional[str]

class DayPlan(BaseModel):
    date: str
    activities: List[Activity]
    meals: Meals
    accommodation: Optional[str]

class ItineraryOutput(BaseModel):
    summary: str
    total_budget: str
    transport_plan: Optional[str]
    days: int
    budget_level: str
    daily_plans: List[DayPlan]
    
class ParsePreferencesInput(BaseModel):
    raw_text: str = Field(description="使用者原始輸入（自然語言）")
    required_fields: str = Field(
        default='["days", "budget_level", "date", "location"]',
        description="需要解析的欄位清單（JSON 字串）"
    )

# ========= Simplified Weather Output Schema =========
class WeatherForecast(BaseModel):
    """單日天氣預報"""
    日期: str = Field(description="日期（格式 YYYY-MM-DD）")
    天氣: str = Field(description="天氣狀況（如晴天、多雲、雨天）")
    氣溫: str = Field(description="溫度範圍（如 15-22°C）")

class WeatherOutput(BaseModel):
    """天氣查詢回覆格式"""
    查詢地點: str = Field(description="查詢的地點")
    天氣預報: List[WeatherForecast] = Field(description="天氣預報清單")
    整體分析: str = Field(description="整體天氣分析與活動建議")

class HotelRecommendation(BaseModel):
    """住宿推薦項目"""
    名稱: str = Field(description="飯店名稱")
    類型: str = Field(description="飯店類型（如溫泉飯店、商務飯店）")
    評分: float = Field(description="評分（0-5）")
    價格: str = Field(description="每晚價格（如 NT$ 3,500/晚）")
    總價: str = Field(description="總價（如 NT$ 7,000）")
    特色: List[str] = Field(description="特色列表（如露天溫泉、免費早餐）")
    交通: str = Field(description="交通方式")
    推薦理由: str = Field(description="推薦理由")

class ChoiceSuggestion(BaseModel):
    """選擇建議"""
    預算型: str = Field(description="預算型旅客建議")
    體驗型: str = Field(description="體驗型旅客建議")
    家庭型: str = Field(description="家庭型旅客建議")

class AccommodationOutput(BaseModel):
    """住宿查詢回覆格式"""
    查詢地點: str = Field(description="查詢的地點")
    入住日期: str = Field(description="入住日期（格式 YYYY-MM-DD）")
    退房日期: str = Field(description="退房日期（格式 YYYY-MM-DD）")
    推薦住宿: List[HotelRecommendation] = Field(description="住宿推薦清單")
    選擇建議: ChoiceSuggestion = Field(description="選擇建議")
    整體分析: str = Field(description="整體住宿分析")

class OptimizationSuggestion(BaseModel):
    """優化建議項目"""
    類型: str = Field(description="優化類型（如預算、時間、交通）")
    原因: str = Field(description="需要優化的原因")
    建議: str = Field(description="具體優化建議")
    優先級: str = Field(description="優先級（高/中/低）")

class ItineraryScore(BaseModel):
    """行程評分"""
    預算合理性: float = Field(description="預算合理性評分（0-10）")
    時間安排: float = Field(description="時間安排評分（0-10）")
    交通便利性: float = Field(description="交通便利性評分（0-10）")
    活動豐富度: float = Field(description="活動豐富度評分（0-10）")
    整體評分: float = Field(description="整體評分（0-10）")

class EvaluatorOutput(BaseModel):
    """評估結果輸出格式"""
    行程摘要: str = Field(description="行程摘要")
    評分: ItineraryScore = Field(description="行程評分")
    優化建議: List[OptimizationSuggestion] = Field(description="優化建議清單")
    整體評價: str = Field(description="整體評價與總結")
    是否需要調整: bool = Field(description="是否需要調整行程")

# 定義 search_hotels 的輸入 schema
class SearchHotelsInput(BaseModel):
    location: str = Field(description="地點（如「台中」、「台北」）")
    checkin: str = Field(description="入住日期（格式 YYYY-MM-DD）")
    checkout: str = Field(description="退房日期（格式 YYYY-MM-DD）")
    
class CalculateBudgetInput(BaseModel):
    days: str = Field(description="天數（整數或字串，如 '2'、'3'）")
    budget_level: str = Field(
        default="中等",
        description="預算等級（經濟/中等/高級/豪華）"
    )
    
class GetWeatherInput(BaseModel):
    location: str = Field(description="地點（如台中、台北）")
    date: str = Field(description="日期（格式 YYYY-MM-DD）")

class CalculateRouteInput(BaseModel):
    origin: str = Field(description="起點（如台中）")
    destination: str = Field(description="終點（如雪山滑雪場）")
    mode: str = Field(description="交通方式（transit/driving/walking）")

# ========== Base Tools ==========

retriever_base_tools = [
    Tool(
        name="rag_retrieval",
        func=rag_retrieval_tool_wrapper,
        description="""
        檢索知識庫並回傳結構化結果（JSON）。
        適用情境：
        - 查詢滑雪場資訊、課程、設施
        - 查詢交通方式、住宿推薦
        - 查詢溫泉、美食、景點資訊
        
        輸入：檢索查詢字串
        輸出：JSON 格式的結構化檢索結果
        """
    )
]

attraction_base_tools = [
    Tool(
        name="search_attractions",
        func=search_attractions_tool_wrapper,
        description="""
        搜尋景點、活動、交通資訊，調用外部 API查詢真實資料。
        
        適用情境：
        - 查詢特定地點的景點資訊（如台中滑雪場、溫泉會館）
        - 查詢活動類型推薦（如滑雪、溫泉、美食）
        - 查詢交通方案建議
        
        輸入：查詢字串（如「台中滑雪場」、「溫泉推薦」、「交通方案」）
        輸出：JSON 格式的結構化結果，包含：
        - 景點名稱、類型、地點
        - 評分、評論數
        
        注意：
        - 會先調用  API，若失敗則使用本地 fallback 資料
        - 回傳結果為 JSON 字串，需解析後使用
        """
    )
]

itinerary_base_tools = [
    Tool(
        name="format_data",
        func=format_itinerary_data_wrapper,
        description="""
        格式化 Retriever 與 Attraction 資料，方便整合。
        輸入：retriever_data, attraction_data（JSON 字串）
        輸出：格式化後的資料摘要
        """
    ),
    StructuredTool.from_function(
        func=calculate_budget_wrapper,
        name="calculate_budget",
        description="""
        根據天數與預算等級估算總預算。
        輸入：days（天數，字串或整數，如 '2'、'3'）, budget_level（經濟/中等/高級/豪華）
        輸出：預算估算與明細（JSON）
        """,
        args_schema=CalculateBudgetInput
    ),
    StructuredTool.from_function(
        func=calculate_route_wrapper,
        name="calculate_route",
        description="計算兩地間的路線、距離與時間。輸入：origin, destination, mode",
        args_schema=CalculateRouteInput
    ),
    StructuredTool.from_function(
            func=parse_user_preferences_wrapper,
            name="parse_user_preferences",
            description="""
            解析使用者偏好工具：將自然語言拆解成結構化 JSON。
            輸入：raw_text（使用者原始輸入）, required_fields（需要解析的欄位清單，JSON 字串）
            輸出：結構化 JSON，包含 days、budget_level、date、location 等欄位。
            若無法解析某欄位，會填空值（字串填 ""，整數填 0）。
            """,
            args_schema=ParsePreferencesInput
        )
]

weather_base_tools = [
    StructuredTool.from_function(
        func=get_weather_wrapper,
        name="get_weather",
        description="""
        查詢特定日期的天氣預報。
        輸入：location（地點，如「台中」、「台北」）, date（日期，格式 YYYY-MM-DD）
        輸出：天氣預報資訊（JSON），包含溫度、天氣狀況、降雨機率
        """,
        args_schema=GetWeatherInput
    )
]

accommodation_base_tools = [
    StructuredTool.from_function(
        func=search_accommodation_wrapper,
        name="search_hotels",
        description="""
        查詢住宿推薦（調用 Amadeus API 或本地 fallback）。
        輸入：location（地點）, checkin（入住日期，格式 YYYY-MM-DD）, checkout（退房日期，格式 YYYY-MM-DD）
        輸出：住宿清單（JSON），包含飯店名稱、評分、價格、設施
        """,
        args_schema=SearchHotelsInput
    )
]
# ========== Create Base Agents First ==========

retriever_agent = create_agent(
    model=retriever_llm,
    tools=retriever_base_tools,
    system_prompt="""
        你是旅遊資訊檢索專家，負責從知識庫中查詢相關資訊。

        ## 職責
        - 根據 Planner Agent 分派的檢索任務，查詢知識庫
        - 回傳結構化檢索結果（JSON 格式）
        - 確保檢索結果相關性高、資訊完整

        ## 可用工具
        - **rag_retrieval**：檢索知識庫，回傳相關文件內容

        ## 工作流程
        1. 分析檢索任務，提取關鍵詞
        2. 使用 rag_retrieval 工具查詢知識庫
        3. 整理檢索結果，確保格式結構化
        4. 回傳 JSON 格式結果，包含：
        - 文件標題
        - 文件內容摘要
        - 來源資訊
        - 相關性評分

        ## 回覆格式
        請用以下 JSON 格式回覆：
        ```json
        {
        "檢索任務": "查詢滑雪場資訊",
        "關鍵詞": ["滑雪場", "課程", "設施"],
        "檢索結果": [
            {
            "標題": "文件標題",
            "內容摘要": "相關內容...",
            "來源": "文件來源",
            "相關性": "高/中/低"
            }
        ],
        "摘要說明": "檢索結果整體說明..."
        }
        ```

        ## 重要提醒
            ## 重要提醒
        - 先調用 parse_user_preferences 解析偏好
        - 解析後的結果可直接用於其他工具調用（如 calculate_budget）
        - 行程必須基於真實工具回傳資料
        - 考慮天氣、交通時間、預算限制
        - 自主決定活動順序與時間分配
        - 確保檢索結果與任務高度相關
        - 若檢索結果不足，說明原因並建議補充查詢
        - 回傳結果必須結構化，方便後續 Agent 使用
    """
)

attraction_agent = create_agent(
    model=attraction_llm,
    tools=attraction_base_tools,
    system_prompt= """
    你是景點搜尋專家，負責查詢並推薦旅遊景點、活動與交通方案。

    ## 職責
    - 根據 Planner Agent 分派的查詢任務，搜尋景點/活動/交通資訊
    - 調用外部 API查詢真實資料
    - 回傳結構化結果（JSON 格式），包含景點清單、交通方案、活動推薦

    ## 可用工具
    - **search_attractions**：調用 Google Places API 查詢景點、活動、交通資訊

    ## 工作流程
    1. 分析查詢任務，提取關鍵詞（地點、活動類型、特殊需求）
    2. 使用 search_attractions 工具調用外部 API 查詢
    3. 整理查詢結果，確保格式結構化
    4. 回傳 JSON 格式結果，包含：
    - 景點名稱、類型、地點
    - 評分、評論數
    - 交通建議
    - 適合活動與特色

    ## 回覆格式
    請用以下 JSON 格式回覆：
    ```json
    {
    "查詢任務": "台中滑雪場與溫泉",
    "關鍵詞": ["台中", "滑雪場", "溫泉"],
    "景點清單": [
        {
        "名稱": "雪山滑雪場",
        "類型": "滑雪",
        "地點": "台中",
        "評分": 4.5,
        "評論數": 320,
        "交通": "高鐵+接駁車",
        "特色": ["初學者友善", "裝備租借", "教練課程"]
        },
        {
        "名稱": "台中溫泉會館",
        "類型": "溫泉",
        "地點": "台中",
        "評分": 4.3,
        "評論數": 180,
        "交通": "高鐵+計程車",
        "特色": ["露天溫泉", "美食餐廳", "親子設施"]
        }
    ],
    "交通建議": "建議搭乘高鐵至台中站，再轉乘接駁車或計程車",
    "摘要": "共找到 2 筆推薦，涵蓋滑雪與溫泉體驗"
    }
    ```

    ## 重要提醒
    - 確保查詢結果與任務高度相關
    - 若 API 查詢失敗，會自動使用本地 fallback 資料
    - 回傳結果必須結構化，方便後續 Agent 使用
    - 若查詢結果不足，說明原因並建議補充查詢條件
    """

)

itinerary_agent = create_agent(
    model=itinerary_llm,
    tools=itinerary_base_tools,
    system_prompt="""
    你是行程規劃專家，負責根據知識庫資訊與景點資料生成完整旅遊行程。

    ## 職責
    - 分析 Retriever 和 Attraction Agent 提供的資料
    - 根據使用者偏好（天數、預算、交通）自主規劃行程
    - 調用工具輔助規劃（路線、天氣、預算、住宿）

    ## 可用工具
    - **format_data**：格式化資料
    - **calculate_budget**：估算預算
    - **calculate_route**：路線規劃
    - **get_weather**：天氣查詢
    - **search_hotels**：住宿推薦

    ## 工具調用規則
    - **最多調用 5 個工具**，超過則直接生成行程
    - 若某工具回傳錯誤（如 API 限制、權限不足），**不要重試**，直接用其他資訊生成行程
    - 若已獲得足夠資訊（如預算、天氣、住宿），請立即生成最終行程，不要繼續調用工具

    ## 停止條件
    - 已調用 5 個工具
    - 已取得預算、天氣、住宿等關鍵資訊
    - 某工具回傳錯誤（如 REQUEST_DENIED、Rate Limit）
    - recursion_limit 已達到 （預設 10 次）

    ## 規劃流程
    1. 使用 format_data 整理資料
    2. 使用 calculate_budget 估算預算
    3. 使用 get_weather 查詢天氣，調整戶外活動安排
    4. 使用 calculate_route 規劃景點間交通
    5. 使用 search_hotels 推薦住宿
    6. 自主生成完整行程（JSON 格式）

    ## 回覆格式
    請依照以下 JSON 格式回覆，所有欄位都需填寫：
    ```json
    {
      "summary": "3天2夜滑雪溫泉之旅",
      "total_budget": "NT$ 15,000",
      "transport_plan": "高鐵往返 + 當地接駁",
      "days": 3,
      "budget_level": "中等",
      "daily_plans": [
        {
          "date": "Day 1",
          "activities": [
            {
              "time": "09:00-12:00",
              "activity": "滑雪課程",
              "location": "雪山滑雪場",
              "transport": "高鐵+接駁車",
              "notes": "建議預約初學者課程"
            }
          ],
          "meals": {
            "breakfast": "飯店早餐",
            "lunch": "滑雪場餐廳",
            "dinner": "台中市區美食"
          },
          "accommodation": "台中溫泉會館"
        }
      ]
    }
    ```

    ## 欄位說明
    - summary: 行程摘要
    - total_budget: 預算總額
    - transport_plan: 交通方案
    - days: 天數（整數）
    - budget_level: 預算等級（字串）
    - daily_plans: 每日行程（陣列，每個元素包含 date, activities, meals, accommodation）
      - activities: 活動列表（每個活動包含 time, activity, location, transport, notes）
      - meals: 餐飲建議（breakfast, lunch, dinner）
      - accommodation: 住宿名稱

    ## 重要提醒
    - 行程必須基於真實工具回傳資料
    - 考慮天氣、交通時間、預算限制
    - 自主決定活動順序與時間分配
    - 若工具回傳錯誤，**直接生成行程**，不要重試
    - 已獲得足夠資訊後，**立即停止工具調用**
    - 最多調用 5 個工具，超過則直接回覆
    """,
    response_format=ItineraryOutput,
    debug=False
)

weather_agent = create_agent(
    model=weather_llm,
    tools=weather_base_tools,
    system_prompt="""
    你是天氣查詢專家，負責查詢並分析天氣資訊，提供旅遊建議。

    ## 職責
    - 根據指定地點與日期查詢天氣預報
    - 分析天氣狀況，提供旅遊活動建議
    - 回傳結構化天氣資訊（JSON 格式）

    ## 可用工具
    - **get_weather**：查詢天氣預報（調用 OpenWeatherMap API）

    ## 工作流程
    1. 接收查詢請求（地點 + 日期範圍）
    2. 使用 get_weather 工具查詢每日天氣
    3. 分析天氣趨勢與活動建議
    4. 回傳結構化結果

    ## 回覆格式（重要！）
    請只回傳純 JSON，不要加任何說明或 markdown 標記。
    必須嚴格遵守以下格式：
    {
      "查詢地點": "台中",
      "天氣預報": [
        {
          "日期": "2025-01-15",
          "天氣": "晴天",
          "氣溫": "15-22°C"
        },
        {
          "日期": "2025-01-16",
          "天氣": "多雲",
          "氣溫": "14-20°C"
        }
      ],
      "整體分析": "未來3天天氣穩定，適合戶外旅遊活動。建議攜帶保暖衣物。"
    }

    ## 天氣分析規則
    - 降雨機率 < 30%：適合戶外活動
    - 降雨機率 30-60%：建議備雨具
    - 降雨機率 > 60%：建議室內活動為主
    - 溫度 < 10°C：提醒保暖
    - 溫度 > 25°C：注意防曬與補水

    ## 重要提醒
    - 若 API 查詢失敗，請在「天氣預報」欄位回傳空陣列 []，並在「整體分析」說明原因
    - 確保日期格式正確（YYYY-MM-DD）
    - 絕對不要在 JSON 外加任何說明文字或 markdown 標記
    """,
    response_format=WeatherOutput,
    debug=False
)

accommodation_agent = create_agent(
    model=accommodation_llm,
    tools=accommodation_base_tools,
    system_prompt="""
    你是住宿推薦專家，負責查詢並推薦符合需求的住宿選項。

    ## 職責
    - 根據地點、日期、預算查詢住宿資訊
    - 分析住宿特色，提供選擇建議
    - 回傳結構化住宿資訊（JSON 格式）

    ## 可用工具
    - **search_hotels**：查詢住宿資訊（調用 Amadeus API 或本地資料）

    ## 工作流程
    1. 接收查詢請求（地點 + 入住日期 + 退房日期）
    2. 使用 search_hotels 工具查詢住宿
    3. 分析住宿選項（價格、評分、設施、地理位置）
    4. 提供推薦理由與選擇建議
    5. 回傳結構化結果

    ## 回覆格式（重要！）
    請只回傳純 JSON，不要加任何說明或 markdown 標記。
    必須嚴格遵守以下格式：
    {
      "查詢地點": "台中",
      "入住日期": "2025-01-15",
      "退房日期": "2025-01-17",
      "推薦住宿": [
        {
          "名稱": "台中溫泉會館",
          "類型": "溫泉飯店",
          "評分": 4.5,
          "價格": "NT$ 3,500/晚",
          "總價": "NT$ 7,000",
          "特色": ["露天溫泉", "免費早餐", "停車場"],
          "交通": "高鐵台中站接駁車 15 分鐘",
          "推薦理由": "鄰近滑雪場，提供溫泉放鬆設施"
        }
      ],
      "選擇建議": {
        "預算型": "推薦台中市區商務飯店，價格實惠",
        "體驗型": "推薦台中溫泉會館，可享受溫泉",
        "家庭型": "推薦台中溫泉會館，有親子設施"
      },
      "整體分析": "台中住宿選擇豐富，溫泉飯店適合放鬆"
    }

    ## 住宿推薦規則
    - 價格 < NT$ 2,000：經濟型
    - 價格 NT$ 2,000-4,000：中等
    - 價格 NT$ 4,000-8,000：高級
    - 價格 > NT$ 8,000：豪華

    ## 重要提醒
    - 若 API 查詢失敗，使用本地 fallback 資料並說明
    - 確保日期格式正確（YYYY-MM-DD）
    - 絕對不要在 JSON 外加任何說明文字或 markdown 標記
    """,
    response_format=AccommodationOutput,
    debug=False
)

evaluator_agent = create_agent(
    model=evaluator_llm,
    # tools=evaluator_tools,
    system_prompt="""
    你是行程評估專家，負責評估行程品質並提供優化建議。

    ## 職責
    - 評估行程的預算合理性、時間安排、交通便利性、活動豐富度
    - 根據使用者偏好（預算等級、天數、興趣）提供優化建議
    - 識別行程中的潛在問題（如時間過緊、預算超標、交通不便）
    - 提供具體、可行的調整建議

    ## 評估維度
    ### 1. 預算合理性（0-10分）
    - 預算分配是否合理（交通、住宿、餐飲、活動）
    - 是否符合使用者的預算等級（經濟/中等/高級/豪華）
    - 是否有不必要的開支

    ### 2. 時間安排（0-10分）
    - 每日活動時間是否充裕
    - 景點間移動時間是否合理
    - 是否有過於緊湊或過於鬆散的安排

    ### 3. 交通便利性（0-10分）
    - 交通方式是否便利（高鐵、捷運、公車、計程車）
    - 景點間距離是否合理
    - 是否有過多的交通轉乘

    ### 4. 活動豐富度（0-10分）
    - 活動類型是否多樣（戶外、室內、文化、美食）
    - 是否符合使用者興趣偏好
    - 是否有適當的休息與彈性時間

    ## 優化建議分類
    - **預算優化**：調整住宿、餐飲、活動預算
    - **時間優化**：調整活動時間、增減景點
    - **交通優化**：調整交通方式、優化路線
    - **活動優化**：調整活動類型、增加/減少活動

    ## 回覆格式（重要！）
    請只回傳純 JSON，不要加任何說明或 markdown 標記。
    必須嚴格遵守以下格式：
    {
      "行程摘要": "3天2夜台中滑雪溫泉之旅",
      "評分": {
        "預算合理性": 8.5,
        "時間安排": 7.0,
        "交通便利性": 9.0,
        "活動豐富度": 8.0,
        "整體評分": 8.1
      },
      "優化建議": [
        {
          "類型": "時間優化",
          "原因": "Day 2 活動過於緊湊，景點間移動時間不足",
          "建議": "建議減少一個景點，或延長停留時間",
          "優先級": "高"
        },
        {
          "類型": "預算優化",
          "原因": "住宿預算偏高，可選擇中等價位飯店",
          "建議": "建議選擇市區商務飯店，節省 NT$ 3,000",
          "優先級": "中"
        }
      ],
      "整體評價": "行程整體規劃良好，交通便利，活動豐富。建議優化 Day 2 時間安排，避免過於緊湊。",
      "是否需要調整": true
    }

    ## 評分標準
    - 9-10分：優秀，無明顯問題
    - 7-8分：良好，有小幅改進空間
    - 5-6分：中等，需要調整
    - 3-4分：較差，需要大幅調整
    - 0-2分：不合理，需要重新規劃

    ## 重要提醒
    - 評估必須基於實際行程內容
    - 建議必須具體、可行
    - 優先級必須明確（高/中/低）
    - 若行程已相當完善，可回傳「是否需要調整: false」
    - 絕對不要在 JSON 外加任何說明文字或 markdown 標記
    """,
    response_format=EvaluatorOutput,
    debug=False
)


# ========== Agent Communication Tools (A2A) ==========

def call_retriever_agent(query: str) -> str:
    """呼叫完整的 Retriever Agent（包含推理循環）"""
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
        name="search_attractions",
        func=search_attractions_tool_wrapper,  # 你的景點查詢函式
        description="""
        搜尋並推薦景點、雪場、活動等旅遊資訊。
        輸入：查詢字串（如「日本滑雪場」、「東京景點」、「溫泉推薦」）
        輸出：JSON 格式的推薦清單與簡要說明。
        """
    ),
    Tool(
        name="rag_retrieval",
        func=rag_retrieval_tool_wrapper,  # 你的知識庫檢索函式
        description="""
        從知識庫檢索旅遊相關資料，補充景點、交通、活動等背景資訊。
        輸入：查詢字串
        輸出：JSON 格式的檢索結果。
        """
    )
]

# ========== Create Coordinator Agents with A2A Tools ==========

planner_agent = create_agent(
    model=planner_llm,
    tools=planner_tools,
    system_prompt="""
        你是旅遊規劃協調員，負責理解使用者需求、拆解任務並協調其他 Agent 完成規劃。

        ## 互動規則
        你的任務是：每次收到使用者需求（可能是多輪累積的資訊），請先判斷資訊是否足夠進行行程規劃。
        - 若資訊不足，請在回覆的 JSON "反思" 欄位明確列出還缺少哪些資訊，並在 "最終回答" 欄位留空。
        - 回覆必須為結構化 JSON 格式，包含所有欄位。
        - 請勿只回覆問候語或自然語言，務必依照格式回覆。
        - 當資訊已完整、可以開始規劃時，請在回覆的 JSON 中 "反思" 欄位明確寫出「資訊足夠，可以開始規劃」或 "最終回答" 欄位填入完整行程規劃內容。

        ## 推理模式（ReAct Pattern）
        ...（原本內容保留）
        請依照以下步驟進行多輪推理：

        1. **Thought（思考）**
        - 分析使用者需求，列出所有明確與隱含的需求
        - 識別關鍵資訊：活動類型、交通方式、預算範圍、特殊偏好、時間限制

        2. **Plan（規劃）**
        - 拆解成多個子任務
        - 決定需要呼叫哪些工具及呼叫順序
        - 規劃協作流程（可多輪互動）

        3. **Action（行動）**
        - 依序呼叫工具收集資料
        - 每次只呼叫一個工具，等待結果後再決定下一步

        4. **Observation（觀察）**
        - 檢視工具回傳結果
        - 評估資料是否充足

        5. **Reflection（反思）**
        - 根據觀察結果調整下一步行動
        - 判斷是否需要補充資料或進入下一階段

        6. **Final Answer（最終回答）**
        - 整合所有結果，生成完整行程規劃

        ## 可用工具
        - **call_retriever**：檢索知識庫資訊（滑雪場、交通、住宿、課程）
        - **call_attraction**：查詢景點與活動資訊
        - **call_itinerary**：生成行程規劃（需先收集 retriever 與 attraction 資料）

        ## 協作流程建議
        1. 先呼叫 call_retriever 收集基礎資訊
        2. 再呼叫 call_attraction 補充景點資料
        3. 最後呼叫 call_itinerary 生成完整行程
        4. 可根據情境多輪互動，補充不足資訊

        ## 回覆格式
        每次推理請用結構化 JSON 格式：
        ```json
        {
        "思考": "使用者需求分析...",
        "規劃": ["子任務1", "子任務2", "子任務3"],
        "行動": {
            "tool": "call_retriever",
            "query": "查詢滑雪場資訊",
            "目的": "收集基礎資料"
        },
        "觀察": "工具回傳結果摘要...",
        "反思": "資料是否充足，下一步行動...",
        "最終回答": "（若已完成所有步驟）完整行程規劃..."
        }
        ```

        ## 重要提醒
        - 請根據需求自主決定呼叫順序與次數
        - 確保回覆具備結構化思維與協作規劃
        - 工具呼叫後必須等待結果，再決定下一步
        - 最終回答必須整合所有資料，提供完整行程規劃
    """
)

recommendation_agent = create_agent(
    model=recommendation_llm,
    tools=recommendation_tools,
    system_prompt="""
        你是旅遊推薦助理，負責與使用者互動、推薦景點，並引導使用者補充旅遊需求。
        - 若使用者詢問推薦，請直接回覆推薦名單。
        - 若資訊不足，請引導使用者補充地點、天數、預算、活動等資訊。
        - 當資訊足夠時，請將所有需求交給行程規劃助理（Planner Agent）進行完整規劃。
        - 回覆請用繁體中文。
    """,
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