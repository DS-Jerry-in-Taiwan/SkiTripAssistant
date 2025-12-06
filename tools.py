from langchain_core.tools import Tool, StructuredTool
from travel_agent_mvp.schemas import (
    SearchHotelsInput, CalculateBudgetInput, GetWeatherInput,
    CalculateRouteInput, ParsePreferencesInput, CallWeatherInput, CallAccommodationInput
)
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
        輸出：JSON 格式的結構化結果，包含景點名稱、類型、地點、評分、評論數
        注意：會先調用 API，若失敗則使用本地 fallback 資料，回傳結果為 JSON 字串，需解析後使用
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
        description="""
        計算兩地間的路線、距離與時間。
        輸入：origin, destination, mode
        輸出：路線、距離、時間等資訊（JSON）
        """,
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
        必須提供：地點（location）、入住日期（checkin）、退房日期（checkout）。
        若缺少任何一項，請主動詢問使用者補充資訊（如：「請輸入入住與退房日期」）。
        輸入：location（地點）, checkin（入住日期，格式 YYYY-MM-DD）, checkout（退房日期，格式 YYYY-MM-DD）
        輸出：住宿清單（JSON），包含飯店名稱、評分、價格、設施
        """,
        args_schema=SearchHotelsInput
    )
]