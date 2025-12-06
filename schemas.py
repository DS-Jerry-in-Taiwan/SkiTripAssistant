from typing import List, Optional
from pydantic import BaseModel, Field

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

class WeatherForecast(BaseModel):
    日期: str = Field(description="日期（格式 YYYY-MM-DD）")
    天氣: str = Field(description="天氣狀況（如晴天、多雲、雨天）")
    氣溫: str = Field(description="溫度範圍（如 15-22°C）")

class WeatherOutput(BaseModel):
    查詢地點: str = Field(description="查詢的地點")
    天氣預報: List[WeatherForecast] = Field(description="天氣預報清單")
    整體分析: str = Field(description="整體天氣分析與活動建議")

class HotelRecommendation(BaseModel):
    名稱: str
    類型: str
    評分: float
    價格: str
    總價: str
    特色: List[str]
    交通: str
    推薦理由: str

class ChoiceSuggestion(BaseModel):
    預算型: str
    體驗型: str
    家庭型: str

class AccommodationOutput(BaseModel):
    查詢地點: str
    入住日期: str
    退房日期: str
    推薦住宿: List[HotelRecommendation]
    選擇建議: ChoiceSuggestion
    整體分析: str

class OptimizationSuggestion(BaseModel):
    類型: str
    原因: str
    建議: str
    優先級: str

class ItineraryScore(BaseModel):
    預算合理性: float
    時間安排: float
    交通便利性: float
    活動豐富度: float
    整體評分: float

class EvaluatorOutput(BaseModel):
    行程摘要: str
    評分: ItineraryScore
    優化建議: List[OptimizationSuggestion]
    整體評價: str
    是否需要調整: bool

class SearchHotelsInput(BaseModel):
    location: str
    checkin: str
    checkout: str

class CalculateBudgetInput(BaseModel):
    days: str
    budget_level: str = "中等"

class GetWeatherInput(BaseModel):
    location: str
    date: str

class CalculateRouteInput(BaseModel):
    origin: str
    destination: str
    mode: str

class CallWeatherInput(BaseModel):
    location: str
    start_date: str
    end_date: Optional[str] = None

class CallAccommodationInput(BaseModel):
    location: str
    checkin: str
    checkout: str