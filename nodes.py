import os
import requests
import tiktoken
import json
import re
from dotenv import load_dotenv
from typing import Dict, Any, List
from datetime import datetime, timedelta
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "travel_knowledge")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
GOOGLE_DIRECTIONS_API_KEY = os.getenv("GOOGLE_DIRECTIONS_API_KEY", "")
SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.7"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))
SYSTEM_PROMPT = (
    "你是一位專業旅遊規劃助理，請務必優先根據下方文件內容，針對使用者問題進行整理、摘要與彙整，不要直接貼出原文。\n"
    "請參考整個對話歷史，維持上下文連貫性，針對使用者持續的需求給出回應。\n"
    "若文件資訊不足，請用旅遊專業知識補充，並說明資料來源非文件庫。\n"
    "請避免重複前一輪回應，針對新問題給出新建議。\n"
    "若已無新資訊可補充，請主動告知使用者，並引導其詢問其他主題。\n"
    "請確保回應簡潔、友善、實用，並使用繁體中文。"
)

# Initialize models
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
vector_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME
)

# ========== Agent 可調用的工具函數 ==========

def load_location_map(filepath: str = None) -> dict:
    """
    載入地名映射表（支援外部 JSON、資料庫或環境變數）
    """
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"載入地名映射表失敗：{e}")
    
    # 預設映射表（可擴充）
    return {
        "台中": "Taichung",
        "台北": "Taipei",
        "高雄": "Kaohsiung",
        "台南": "Tainan",
        "新竹": "Hsinchu",
        "嘉義": "Chiayi",
        "彰化": "Changhua",
        "屏東": "Pingtung",
        "花蓮": "Hualien",
        "台東": "Taitung",
        "宜蘭": "Yilan",
        "桃園": "Taoyuan",
        "苗栗": "Miaoli",
        "南投": "Nantou",
        "雲林": "Yunlin",
        "基隆": "Keelung",
        "新北": "New Taipei City",
        "新潟": "Niigata",
        "東京": "Tokyo",
        "大阪": "Osaka"
    }


def location_standardizer_llm(location: str, model_name: str = "gpt-3.5-turbo") -> str:
    """
    使用 LLM 進行地名語意解析與標準化
    適用於複雜或非標準輸入（如「台中市區」、「台北車站附近」）
    """
    prompt = f"""
    請將以下地名標準化為「城市名稱」（繁體中文）。
    
    規則：
    - 若輸入包含「市區」、「車站」、「附近」等修飾詞，請提取主要城市名稱
    - 若輸入為景點名稱（如「台中溫泉會館」、「湯澤滑雪場」），請提取所在城市
    - 只回傳城市名稱，不要加任何說明或標點符號
    
    範例：
    輸入：「台中市區」 → 輸出：台中
    輸入：「台北車站附近」 → 輸出：台北
    輸入：「台中溫泉會館」 → 輸出：台中
    輸入：「湯澤滑雪場」 → 輸出：新潟
    
    輸入：{location}
    輸出：
    """
    
    try:
        result = generate_prompt(location, prompt, model_name)
        # 清理回應（移除多餘空白、標點）
        standardized = result.strip().replace("「", "").replace("」", "").replace("。", "").replace("\n", "")
        return standardized
    except Exception as e:
        print(f"LLM 地名解析失敗：{e}")
        return location


def location_standardizer_with_detail(location: str, target_type: str = "full", model_name: str = "gpt-4o") -> dict:
    """
    進階地名標準化工具：支援細部地點解析
    
    Args:
        location: 原始地名（如「湯澤滑雪場」、「台中溫泉會館」）
        target_type: 目標類型（"city" 城市級別，"detail" 細部地點，"coords" 經緯度，"full" 完整資訊）
        model_name: LLM 模型名稱
    
    Returns:
        若 target_type="full"，回傳 dict:
        {
            "original": "湯澤滑雪場",
            "city_zh": "新潟",
            "city_en": "Niigata",
            "detail_zh": "湯澤滑雪場",
            "detail_en": "Gala Yuzawa Snow Resort",
            "coords": {"lat": 36.9167, "lng": 138.8333}
        }
        否則回傳對應欄位的值（字串或 dict）
    """
    prompt = f"""
    請解析以下地點資訊，回覆 JSON 格式（不要加 markdown 標記或任何說明文字）。
    
    規則：
    - city_zh: 所在城市（繁體中文，如「新潟」、「台中」）
    - city_en: 所在城市（英文，如「Niigata」、「Taichung」）
    - detail_zh: 細部地點（繁體中文，保持原名，如「湯澤滑雪場」、「台中溫泉會館」）
    - detail_en: 細部地點（英文，如「Gala Yuzawa Snow Resort」、「Taichung Hot Spring Resort」）
    - coords: 經緯度（若已知填 {{"lat": 緯度, "lng": 經度}}，若不確定填 null）
    
    範例：
    輸入：湯澤滑雪場
    輸出：
    {{
      "city_zh": "新潟",
      "city_en": "Niigata",
      "detail_zh": "湯澤滑雪場",
      "detail_en": "Gala Yuzawa Snow Resort",
      "coords": {{"lat": 36.9167, "lng": 138.8333}}
    }}
    
    輸入：台中溫泉會館
    輸出：
    {{
      "city_zh": "台中",
      "city_en": "Taichung",
      "detail_zh": "台中溫泉會館",
      "detail_en": "Taichung Hot Spring Resort",
      "coords": null
    }}
    
    輸入：陽明山國家公園
    輸出：
    {{
      "city_zh": "台北",
      "city_en": "Taipei",
      "detail_zh": "陽明山國家公園",
      "detail_en": "Yangmingshan National Park",
      "coords": {{"lat": 25.1825, "lng": 121.5637}}
    }}
    
    現在請解析：
    輸入：{location}
    輸出：
    """
    
    try:
        result = generate_prompt(location, prompt, model_name)
        # 清理回應（移除 markdown 標記）
        result = result.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(result)
        
        # 補充原始輸入
        parsed["original"] = location
        
        # 根據 target_type 回傳對應資訊
        if target_type == "city":
            return parsed.get("city_en", location)
        elif target_type == "detail":
            return parsed.get("detail_en", location)
        elif target_type == "coords":
            return parsed.get("coords", None)
        else:  # "full"
            return parsed
    
    except Exception as e:
        print(f"地名解析失敗：{e}")
        # Fallback：查詢本地映射表
        location_map = load_location_map()
        return location_map.get(location, location)


def location_standardizer(location: str, target_language: str = "en", use_llm: bool = True) -> str:
    """
    地名標準化工具：將中文地名轉換為標準格式（英文或城市代碼）
    
    Args:
        location: 原始地名（中文或英文）
        target_language: 目標語言（"en" 英文，"zh" 繁體中文，"code" 城市代碼，"detail_en" 細部地點英文）
        use_llm: 是否使用 LLM 進行語意解析（處理複雜輸入）
    
    Returns:
        標準化後的地名
    
    Examples:
        >>> location_standardizer("湯澤滑雪場", target_language="en")
        "Niigata"
        
        >>> location_standardizer("湯澤滑雪場", target_language="detail_en")
        "Gala Yuzawa Snow Resort"
        
        >>> location_standardizer("台中市區", target_language="en")
        "Taichung"
    """
    # 載入地名映射表
    location_map = load_location_map()
    
    # 若輸入已是英文，直接返回
    if location.isascii():
        return location
    
    # 1. 先用 LLM 進行語意解析（處理「台中市區」、「台北車站附近」、「湯澤滑雪場」等）
    if use_llm:
        # 判斷是否為細部地點（包含「場」、「館」、「店」、「區」等關鍵字）
        is_detail = any(keyword in location for keyword in ["場", "館", "店", "區", "山", "公園", "景點", "溫泉", "滑雪"])
        
        if is_detail:
            # 使用進階解析，提取城市級別地名或細部地點
            parsed = location_standardizer_with_detail(location, target_type="full")
            if isinstance(parsed, dict):
                if target_language == "en":
                    # 天氣查詢、住宿查詢：回傳城市級別
                    return parsed.get("city_en", location)
                elif target_language == "detail_en":
                    # 路線規劃、景點查詢：回傳細部地點英文
                    return parsed.get("detail_en", location)
                elif target_language == "zh":
                    return parsed.get("city_zh", location)
                elif target_language == "coords":
                    return parsed.get("coords", None)
        else:
            # 使用簡單解析（如「台中市區」→「台中」）
            location = location_standardizer_llm(location)
    
    # 2. 查詢映射表
    if target_language == "en":
        standardized = location_map.get(location, location)
    elif target_language == "zh":
        standardized = location
    elif target_language == "code":
        # 城市代碼映射（可擴充）
        code_map = {
            "台中": "TXG",
            "台北": "TPE",
            "高雄": "KHH",
            "新潟": "KIJ",
            "東京": "TYO"
        }
        standardized = code_map.get(location, location)
    else:
        standardized = location
    
    # 3. 若無法映射，返回原始輸入
    if standardized == location and target_language == "en":
        print(f"警告：地名 '{location}' 無法映射為英文，使用原始輸入")
    
    return standardized


def location_standardizer_wrapper(location: str, target_language: str = "en") -> str:
    """
    Tool wrapper for Agent - 回傳標準化地名
    """
    return location_standardizer(location, target_language, use_llm=True)

def search_attractions_node(query: str, top_k: int = 5, api_key: str=GOOGLE_PLACES_API_KEY) -> Dict[str, Any]:
    """
    景點搜尋工具 - 調用外部 API 查詢景點資訊
    輸入：查詢字串（如地點、活動、交通）
    輸出：結構化景點清單、交通方案、活動推薦
    """
    results: List[Dict[str, Any]] = []
    
    try:
        # 使用 Google Places API（需申請 API Key）
        api_key = os.getenv("GOOGLE_PLACES_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_PLACES_API_KEY 未設定")
        
        # Google Places API - Text Search
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {
            "query": query,
            "key": api_key,
            "language": "zh-TW"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK":
            for place in data.get("results", [])[:top_k]:
                results.append({
                    "name": place.get("name", "未知景點"),
                    "type": ", ".join(place.get("types", [])),
                    "location": place.get("formatted_address", ""),
                    "rating": place.get("rating", None),
                    "user_ratings_total": place.get("user_ratings_total", 0),
                    "place_id": place.get("place_id", ""),
                    "photo_reference": place.get("photos", [{}])[0].get("photo_reference", "") if place.get("photos") else ""
                })
        else:
            return {
                "query": query,
                "source": "api_error",
                "results": [],
                "error": f"API 錯誤：{data.get('status')}"
            }
    
    except Exception as e:
        # Fallback: 若 API 失敗，嘗試從本地 JSON 或資料庫查詢
        fallback_file = os.path.join(os.path.dirname(__file__), "data", "attractions.json")
        if os.path.exists(fallback_file):
            try:
                with open(fallback_file, "r", encoding="utf-8") as f:
                    fallback_data = json.load(f)
                    # 簡單關鍵字匹配
                    for item in fallback_data:
                        if query.lower() in item.get("name", "").lower() or query.lower() in item.get("type", "").lower():
                            results.append(item)
                            if len(results) >= top_k:
                                break
            except Exception:
                pass
        
        if not results:
            return {
                "query": query,
                "source": "error",
                "results": [],
                "error": f"API 查詢失敗：{str(e)}"
            }
    
    return {
        "query": query,
        "source": "google_places_api",
        "results": results[:top_k],
        "summary": f"共找到 {len(results)} 筆景點/活動推薦"
    }


def search_attractions_tool_wrapper(query: str) -> str:
    """
    Tool wrapper for create_agent.Tool - 回傳 JSON 字串
    """
    result = search_attractions_node(query)
    return json.dumps(result, ensure_ascii=False, indent=2)


def rag_retrieval_node(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    結構化 RAG 檢索工具
    支援 vector store 與 fallback，回傳標準化 dict
    """
    results: List[Dict[str, Any]] = []
    source = "vector"
    try:
        results_with_score = vector_store.similarity_search_with_score(query, k=top_k) if vector_store else []
        filtered_docs_scores = [(doc, score) for doc, score in results_with_score if score <= SCORE_THRESHOLD]
        for idx, (doc, score) in enumerate(filtered_docs_scores):
            results.append({
                "id": getattr(doc, "id", f"doc_{idx}"),
                "title": doc.metadata.get("title", f"文件{idx+1}") if hasattr(doc, "metadata") else f"文件{idx+1}",
                "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                "source": doc.metadata.get("source", "vector_store") if hasattr(doc, "metadata") else "vector_store",
                "score": float(score),
                "metadata": doc.metadata if hasattr(doc, "metadata") else {}
            })
    except Exception as e:
        # fallback: 檔案掃描
        import os, glob
        source = "kb_fallback"
        kb_dir = os.path.join(os.path.dirname(__file__), "kb")
        files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))[:top_k]
        for idx, path in enumerate(files):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                if query.lower() in content.lower():
                    snippet = content[:600].replace("\n", " ")
                    results.append({
                        "id": f"kb_{idx}",
                        "title": os.path.basename(path),
                        "content": snippet,
                        "source": path,
                        "score": None,
                        "metadata": {"file_path": path}
                    })
            except Exception:
                continue

    return {
        "query": query,
        "source": source,
        "results": results[:top_k]
    }

def rag_retrieval_tool_wrapper(query: str) -> str:
    """
    Tool wrapper for create_agent.Tool - 回傳 JSON 字串
    """
    result = rag_retrieval_node(query)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ========== 工具 1：格式化資料 ==========
def format_itinerary_data(retriever_data: str, attraction_data: str) -> str:
    """
    格式化工具：整理 Retriever 與 Attraction 資料，方便 LLM 理解
    """
    try:
        retriever = json.loads(retriever_data) if isinstance(retriever_data, str) else retriever_data
        attraction = json.loads(attraction_data) if isinstance(attraction_data, str) else attraction_data
    except:
        return json.dumps({"error": "資料格式錯誤"}, ensure_ascii=False)
    
    formatted = {
        "知識庫資訊": retriever.get("results", []),
        "景點資訊": attraction.get("results", []),
        "摘要": f"共 {len(retriever.get('results', []))} 筆知識庫資料，{len(attraction.get('results', []))} 個景點"
    }
    
    return json.dumps(formatted, ensure_ascii=False, indent=2)


# ========== 工具 2：預算估算 ==========
def calculate_budget_estimate(days: int, budget_level: str = "中等") -> str:
    """
    預算估算工具：根據天數與預算等級估算總預算
    """
    budget_map = {
        "經濟": 3000,
        "中等": 5000,
        "高級": 10000
    }
    daily_budget = budget_map.get(budget_level, 5000)
    total = days * daily_budget
    
    return json.dumps({
        "天數": days,
        "預算等級": budget_level,
        "每日預算": f"NT$ {daily_budget}",
        "總預算估算": f"NT$ {total}",
        "預算明細": {
            "住宿": f"NT$ {int(daily_budget * 0.4 * days)}",
            "餐飲": f"NT$ {int(daily_budget * 0.3 * days)}",
            "交通": f"NT$ {int(daily_budget * 0.2 * days)}",
            "活動": f"NT$ {int(daily_budget * 0.1 * days)}"
        }
    }, ensure_ascii=False, indent=2)


# ========== 工具 3：路線規劃（Google Maps API）==========
def calculate_route(origin: str, destination: str, mode: str = "transit") -> str:
    """
    路線規劃工具：調用 Google Maps Directions API 計算路線與時間
    """
    api_key = os.getenv("GOOGLE_DIRECTIONS_API_KEY")
    if not api_key:
        return json.dumps({"error": "GOOGLE_DIRECTIONS_API_KEY 未設定"}, ensure_ascii=False)
    
    try:
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "mode": mode,  # transit, driving, walking, bicycling
            "language": "zh-TW",
            "key": api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK":
            route = data["routes"][0]["legs"][0]
            return json.dumps({
                "起點": origin,
                "終點": destination,
                "交通方式": mode,
                "距離": route["distance"]["text"],
                "預估時間": route["duration"]["text"],
                "詳細路線": route["steps"][0]["html_instructions"] if route.get("steps") else "無"
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({"error": f"路線查詢失敗：{data.get('status')}"}, ensure_ascii=False)
    
    except Exception as e:
        return json.dumps({"error": f"API 調用失敗：{str(e)}"}, ensure_ascii=False)


# ========== 工具 4：天氣查詢（OpenWeatherMap API）==========
def get_weather_forecast(location: str, date: str) -> str:
    """
    天氣查詢工具：調用 OpenWeatherMap API 查詢天氣預報
    """
    location_en = location_standardizer(location, target_language="en", use_llm=True)
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return json.dumps({"error": "OPENWEATHER_API_KEY 未設定"}, ensure_ascii=False)
    
    try:
        # 使用 5-day forecast API
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "q": location_en,
            "appid": api_key,
            "units": "metric",
            "lang": "zh_tw"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 找出最接近目標日期的預報
        target_date = datetime.strptime(date, "%Y-%m-%d")
        forecasts = []
        
        for item in data.get("list", []):
            forecast_time = datetime.fromtimestamp(item["dt"])
            if forecast_time.date() == target_date.date():
                forecasts.append({
                    "時間": forecast_time.strftime("%H:%M"),
                    "天氣": item["weather"][0]["description"],
                    "溫度": f"{item['main']['temp']}°C",
                    "降雨機率": f"{item.get('pop', 0) * 100}%"
                })
        
        return json.dumps({
            "地點": location,
            "日期": date,
            "預報": forecasts[:4] if forecasts else [{"提示": "無預報資料"}]
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        return json.dumps({"error": f"天氣查詢失敗：{str(e)}"}, ensure_ascii=False)


# ========== 工具 5：住宿查詢（Booking.com API - 可選）==========
def search_accommodation_fallback(location: str, checkin: str, checkout: str) -> str:
    """
    Fallback：使用本地住宿資料
    """
    # 從 JSON 檔案或資料庫查詢
    fallback_file = os.path.join(os.path.dirname(__file__), "data", "accommodations.json")
    
    if os.path.exists(fallback_file):
        try:
            with open(fallback_file, "r", encoding="utf-8") as f:
                all_hotels = json.load(f)
            
            # 簡單關鍵字匹配
            hotels = [h for h in all_hotels if location.lower() in h.get("location", "").lower()][:5]
        except:
            hotels = []
    else:
        hotels = [
            {
                "name": "台中溫泉會館",
                "rating": 4.5,
                "price": "NT$ 3,500/晚",
                "features": ["溫泉", "早餐", "免費停車"]
            },
            {
                "name": "市區商務飯店",
                "rating": 4.2,
                "price": "NT$ 2,200/晚",
                "features": ["近捷運", "健身房", "商務中心"]
            }
        ]
    
    return json.dumps({
        "地點": location,
        "入住日期": checkin,
        "退房日期": checkout,
        "推薦住宿": hotels,
        "來源": "本地資料"
    }, ensure_ascii=False, indent=2)

def search_accommodation(location: str, checkin: str, checkout: str) -> str:
    """
    使用 Amadeus Hotel Search API 查詢住宿
    """
    # 1. 先取得 access token
    client_id = os.getenv("AMADEUS_API_KEY")
    client_secret = os.getenv("AMADEUS_API_SECRET")
    
    if not client_id or not client_secret:
        return json.dumps({"error": "Amadeus API credentials 未設定"}, ensure_ascii=False)
    
    try:
        # 取得 token
        auth_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        auth_data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret
        }
        auth_response = requests.post(auth_url, data=auth_data, timeout=10)
        auth_response.raise_for_status()
        access_token = auth_response.json()["access_token"]
        
        # 2. 查詢飯店（先用 cityCode，實際使用需先查詢城市代碼）
        # 這裡簡化處理，實際應先調用 Hotel List API 取得 hotelIds
        search_url = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {
            "cityCode": "TPE",  # 台北，實際需根據 location 查詢
            "checkInDate": checkin,
            "checkOutDate": checkout,
            "adults": 2,
            "radius": 50,
            "radiusUnit": "KM"
        }
        
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hotels = []
        for offer in data.get("data", [])[:5]:
            hotel = offer.get("hotel", {})
            price_info = offer.get("offers", [{}])[0].get("price", {})
            
            hotels.append({
                "name": hotel.get("name", "未知飯店"),
                "rating": hotel.get("rating", "N/A"),
                "price": f"{price_info.get('currency', 'TWD')} {price_info.get('total', 'N/A')}",
                "features": [hotel.get("amenities", ["無資料"])[0]] if hotel.get("amenities") else []
            })
        
        return json.dumps({
            "地點": location,
            "入住日期": checkin,
            "退房日期": checkout,
            "推薦住宿": hotels,
            "來源": "Amadeus API"
        }, ensure_ascii=False, indent=2)
    
    except Exception as e:
        # Fallback 到本地資料
        return search_accommodation_fallback(location, checkin, checkout)

def parse_user_preferences_tool(raw_text: str, required_fields: str = "") -> str:
    """
    解析使用者偏好工具：將自然語言拆解成結構化 JSON
    
    Args:
        raw_text: 原始使用者輸入（自然語言）
        required_fields: 需要解析的欄位清單（JSON 字串），例如 '["days", "budget_level", "date", "location"]'
    
    Returns:
        結構化 JSON 字串，包含解析結果
    """
    # 解析 required_fields
    try:
        fields_list = json.loads(required_fields) if required_fields else []
    except:
        fields_list = ["days", "budget_level", "date", "location"]
    
    # 構建 prompt
    fields_desc = ", ".join(fields_list)
    prompt = f"""
    請將以下旅遊偏好資訊精確拆解成 JSON 格式，欄位包括：{fields_desc}。
    
    ## 欄位解析規則：
    - **days**（必須是整數）：
      * 從「2天1夜」中提取數字「2」
      * 從「3天2夜」中提取數字「3」
      * 從「玩2天」中提取數字「2」
      * 只回傳天數的數字，不要包含「天」、「夜」等文字
      * 若無法解析請填 0
    
    - **budget_level**（字串）：
      * 可能的值：「經濟」、「中等」、「高級」、「豪華」
      * 從「預算中等」中提取「中等」
      * 若無法解析請填空字串 ""
    
    - **date**（字串，格式 YYYY-MM-DD）：
      * 從「出發日期是2024-12-20」中提取「2024-12-20」
      * 從「12月20日出發」轉換為「2024-12-20」
      * 必須是標準日期格式 YYYY-MM-DD
      * 若無法解析請填空字串 ""
    
    - **location**（字串）：
      * 從「我想去台中玩」中提取「台中」
      * 從「想去台北溫泉」中提取「台北」
      * 只回傳地點名稱
      * 若無法解析請填空字串 ""
    
    ## 解析範例：
    輸入：「我想去台中玩2天1夜，預算中等，出發日期是2024-12-20」
    正確輸出：
    {{
      "days": 2,
      "budget_level": "中等",
      "date": "2024-12-20",
      "location": "台中"
    }}
    
    輸入：「預計玩3天2夜，預算高級」
    正確輸出：
    {{
      "days": 3,
      "budget_level": "高級",
      "date": "",
      "location": ""
    }}
    
    ## 重要提醒：
    - days 欄位必須是純數字（整數型別），不可包含任何文字
    - 若某欄位無法從原始輸入中解析出來，請填預設空值（整數填 0，字串填 ""）
    - 請直接回覆 JSON 格式，不要加任何自然語言說明或 markdown 標記
    - 確保 JSON 格式正確，可被程式直接解析
    
    ## 原始輸入：
    {raw_text}
    
    請回覆標準 JSON 格式：
    """
    
    try:
        # 調用 LLM 解析
        result = generate_prompt(raw_text, prompt, model_name="gpt-4o")
        
        # 清理回應（移除 markdown 標記）
        result = result.replace("```json", "").replace("```", "").strip()
        
        # 驗證 JSON 格式
        parsed = json.loads(result)
        
        # 特別處理 days 欄位，確保是整數
        if "days" in parsed:
            days_value = parsed["days"]
            if isinstance(days_value, str):
                # 從字串中提取數字
                match = re.search(r'(\d+)', days_value)
                parsed["days"] = int(match.group(1)) if match else 0
            elif not isinstance(days_value, int):
                parsed["days"] = 0
        
        # 確保必要欄位存在，缺失則填預設值
        default_values = {
            "days": 0,
            "budget_level": "",
            "date": "",
            "location": ""
        }
        
        for field in fields_list:
            if field not in parsed:
                parsed[field] = default_values.get(field, "")
        
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    
    except Exception as e:
        print(f"解析失敗：{e}")
        # 容錯處理：回傳空值
        default_result = {field: (0 if field == "days" else "") for field in fields_list}
        return json.dumps(default_result, ensure_ascii=False, indent=2)


# ========== Tool Wrappers ==========
def format_itinerary_data_wrapper(retriever_data: str, attraction_data: str) -> str:
    return format_itinerary_data(retriever_data, attraction_data)

def calculate_budget_wrapper(days: str, budget_level: str = "中等") -> str:
    print("=="*100)
    print(f"Calculating budget for {days} days at {budget_level} level.")
    return calculate_budget_estimate(int(days), budget_level)

def calculate_route_wrapper(origin: str, destination: str, mode: str = "transit") -> str:
    return calculate_route(origin, destination, mode)

def get_weather_wrapper(location: str, date: str) -> str:
    return get_weather_forecast(location, date)

def search_accommodation_wrapper(location: str, checkin: str, checkout: str) -> str:
    return search_accommodation(location, checkin, checkout)

def parse_user_preferences_wrapper(raw_text: str, required_fields: str = "") -> str:
    """
    Tool wrapper for Agent - 回傳 JSON 字串
    """
    return parse_user_preferences_tool(raw_text, required_fields)


# ========== 輔助函數 ==========
# retriever setup
RETRIEVER = None
if vector_store:
    RETRIEVER = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVAL_K}
    )
    print("Retriever successfully created.")    
else:
    print("Vector store is not initialized; retriever cannot be created.")

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """ Count the number of tokens in a given text for a specified model. """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_model_token_limit(model_name:str) -> int:
    """Get the token limit for a specified model"""
    model_token_limits = {
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
    }
    return model_token_limits.get(model_name, 4096)

def format_user_preferences(user_prefs: dict) -> str:
    """將偏好 dict 格式化為字串"""
    return ", ".join([f"{k}:{v}" for k, v in user_prefs.items()])

def get_recent_messages(messages: list, role: str = None, n: int = 10) -> str:
    """Get the most recent n user messages from the conversation history"""
    if role:
        filtered = [msg for msg in messages if msg.get("role") == role]
    else:
        filtered = messages
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in filtered[-n:]])

def call_llm_for_json(content: str, prompt: str, model_name: str = "gpt-3.5-turbo") -> Any:
    """ Call LLM to generate JSON output based on the given content and prompt and parse the response"""
    result = generate_prompt(content, prompt, model_name)
    try:
        import json
        return json.loads(result)
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "raw_response": result}

# def call_search_tool(state: Dict[str, Any], api_key: str = TAVILY_API_KEY) -> dict:
#     """ Call the search tool and update the state with results """
#     query = state.get("query", "")
#     location = state.get("user_preferences", {}).get("location", "")
#     state.setdefault("tool_results", {})
#     results = update_acctraction_results(state, query, location,api_key=api_key)
#     return results
def call_initerary_node(state: Dict[str, Any], model_name: str = "gpt-4o") -> Dict[str, Any]:
    """ Call the itinerary generation tool and update the state with results """
    

# def rag_retrieval_node(state: Dict[str, Any], threshold: float = SCORE_THRESHOLD) -> Dict[str, Any]:
#     """ 
#     RAG Retrieval Node
#     Retrieves relevant documents using the pre-configured retriever.
#     """
#     # Extract the latest user message as the query
#     user_message = [msg for msg in state.get("messages", []) if msg.get("role") == "user"]
#     if not user_message:
#         return {"retrieved_docs": []}
#     query = user_message[-1].get("content", "").strip()
#     if not query:
#         return {"retrieved_docs": []}
    
#     # operate retrieve from the vector store
#     try:
#         results_with_score = vector_store.similarity_search_with_score(query, k=RETRIEVAL_K) if vector_store else []
#         # 打印分數
#         # for i, (doc, score) in enumerate(results_with_score, 1):
#         #     print(f"結果 {i}: 分數={score:.4f}, 內容={doc.page_content[:40]}...")
#         filtered_docs_scores = [(doc, score) for doc, score in results_with_score if score <= threshold]
#         filtered_docs = [doc for doc, score in filtered_docs_scores]
#         # return {"retrieved_docs": filtered_docs}
#         # no result handling
#         if not filtered_docs:
#             return {"retrieved_docs": []}
#         return {"retrieved_docs": filtered_docs,
#                 "retrieved_docs_with_score": filtered_docs_scores
#                 }
#     except Exception as e:
#         print(f"Error during retrieval: {e}")
#         return {"retrieved_docs": []}

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_model_token_limit(model_name: str = "gpt-3.5-turbo") -> int:
    model_token_limits = {
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
    }
    return model_token_limits.get(model_name, 4096)

def truncate_prompt_by_token(prompt_messages: list, model_name: str = "gpt-3.5-turbo") -> list:
    token_limit = get_model_token_limit(model_name)
    total_tokens = 0
    truncated = []
    # 順序保留 system、文件資訊
    for msg in prompt_messages[:2]:
        msg_tokens = count_tokens(msg["content"], model_name)
        total_tokens += msg_tokens
        truncated.append(msg)
    # 倒序加入歷史訊息，直到達到限制
    history = prompt_messages[2:]
    for msg in reversed(history):
        msg_tokens = count_tokens(msg["content"], model_name)
        if total_tokens + msg_tokens > token_limit:
            truncated.insert(2, {"role": "system", "content": "（部分對話已省略）"})
            break
        truncated.insert(2, msg)
        total_tokens += msg_tokens
    return truncated[:2] + list(reversed(truncated[2:]))

def format_retrieval_node(state: Dict[str, Any], max_length: int = 1200,top_k: int = 5) -> Dict[str, Any]:
    """
    format retrieved documents in state. Add title and handling too long content.
    """
    docs_with_score = state.get("retrieved_docs_with_score", [])
    if not docs_with_score and state.get("retrieved_docs",[]):
        # if no score info, create dummy scores
        docs_with_score = [(doc, 1.0) for doc in state.get("retrieved_docs",[])]
    if not docs_with_score:
        return "（目前無相關文件，請用通用知識回答）"
    # Accendding sort by score
    sorted_docs = sorted(docs_with_score, key=lambda x: x[1])
    formatted_list = []
    total_length = 0
    for idx, (doc, score) in enumerate(sorted_docs[:top_k], 1):
        title = doc.metadata.get("title", f"document{idx}")
        content = doc.page_content.strip()
        # Handle too long content
        if len(content) > 400:
            content = content[:400] + "...content too long, truncated."
        doc_text = f"【{title}】(score={score:.3f})\n{content}\n---"
        if total_length + len(doc_text) > max_length:
            formatted_list.append("...additional content truncated due to length limit.")
            break
        formatted_list.append(doc_text)
        total_length += len(doc_text)
        
    return "\n".join(formatted_list)

def generate_prompt(content: str, prompt: str = "請摘要以下內容", model_name: str = "gpt-3.5-turbo") -> str:
    """
    generate summary based on the given content and prompt
    """
    llm = ChatOpenAI(model=model_name, openai_api_key=OPENAI_API_KEY, temperature=0.0)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": content}
    ]
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)

def generate_conversation_summary(state: Dict[str, Any], model_name: str = 'gpt-3.5-turbo') -> str:
    """
    generate conversation summary from the state messages
    """
    history_text = get_recent_messages(state.get("messages", []))
    prefs_text = format_user_preferences(state.get("user_preferences", {}))
    prompt = "請根據使用者偏好與對話歷史，摘要目前使用者的旅遊需求與狀態，回覆繁體中文："
    content = f"使用者偏好：{prefs_text}\n對話歷史：\n{history_text}"
    return generate_prompt(content, prompt, model_name)

def update_user_prference(state: Dict[str, Any], model_name: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """retrieve and update user preference in the state"""    
    history_text = get_recent_messages(state.get("messages", []))
    prefs_text = format_user_preferences(state.get("user_prefereneces", {}))
    prompt = (
        "請根據下方對話歷史，更新使用者的旅遊偏好設定，"
        "以JSON格式回覆，欄位包括：預算(budget)、旅遊日期(travel_dates)、活動強度(activity_level)、"
        "目的地偏好(destination_preferences)、住宿偏好(accommodation_preferences)、其他需求(other_requirements)。"
        "若無相關資訊，請保持原設定不變。"
        f"現有偏好：{prefs_text}\n對話歷史：\n{history_text}"
    )
    prefs = call_llm_for_json("", prompt, model_name)
    if isinstance(prefs, dict):
        state["user_preferences"].update(prefs)
    return state
    
def build_prompt(state: Dict[str, Any]) -> list:
    """
    merge system prompt and chat history to build the prompt for LLM
    """
    docs_text = format_retrieval_node(state)
    last_user_msg = ""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
    user_focus = f"本輪問題重點：{last_user_msg}" if last_user_msg else ""
    summary = state.get("conversation_summary", "")
    summary_text = f"對話摘要: {summary}\n" if summary else ""
    # include recent user messages for context
    recent_msgs = [msg["content"] for msg in state.get("messages", []) if msg.get("role") == "user"][-2:]
    recent_context = "對話摘要：" + " / ".join(recent_msgs) if recent_msgs else ""
    prefs_text = format_user_preferences(state.get("user_preferences", {}))
    prefs_block = f"使用者偏好：{prefs_text}\n" if prefs_text else ""
    prompt_user_content = (
        f"以下是參考文件：\n{docs_text}\n"
        f"{summary_text}\n"
        f"{user_focus}\n"
        f"{prefs_block}\n"
        f"近期使用者訊息：{recent_context}"
    )
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_user_content}
    ]
    # add full chat history (user/assistant), latest message at the end
    prompt_messages.extend(state.get("messages", []))
    return prompt_messages

def is_semantically_duplicate(text1: str, text2: str, threshold: float = 0.9) -> bool:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    vec1 = embeddings.embed_query(text1)
    vec2 = embeddings.embed_query(text2)
    # 計算 cosine similarity
    from numpy import dot
    from numpy.linalg import norm
    similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return similarity > threshold


def call_model_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call LLM model to generate response based on the built prompt
    """
    try:
        # merge system prompt and the formated retrieved documnets
        prompt_messages = build_prompt(state)
        prompt_messages = truncate_prompt_by_token(prompt_messages, model_name="gpt-3.5-turbo")
         
        # Initialize the LLM model
        llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.2)

        # Call the model
        response = llm.invoke(prompt_messages)
        ai_content = response.content if hasattr(response, "content") else str(response)
        
        # check for semantic duplication with last assistant message
        last_assistant_msg = ""
        for msg in reversed(state["messages"]):
            if msg["role"] == "assistant":
                last_assistant_msg = msg["content"]
                break
        # 語意重複判斷
        is_duplicate = False
        if last_assistant_msg:
            is_duplicate = is_semantically_duplicate(ai_content.strip(), last_assistant_msg.strip())

        if is_duplicate:
            # 調用 LLM 產生通用回應並引導切換主題，補充使用者偏好
            print("Detected semantically duplicate response. Generating alternative response.")
            context = get_recent_messages(state.get("messages", []), n=6)
            prefs_text = format_user_preferences(state.get("user_preferences", {}))
            prompt = (
                "目前您的問題已獲得完整回答，請根據現有對話情境與使用者偏好："
                f"{prefs_text}\n"
                "提供一段通用回應並主動引導使用者切換到其他旅遊主題或提出新需求，回覆繁體中文。"
            )
            ai_content = generate_prompt(context, prompt, model_name="gpt-3.5-turbo")
            
        # update state with the assistant message
        state["messages"].append({"role": "assistant", "content": ai_content})
        
        # update conversation summary
        summary = generate_conversation_summary(state, model_name="gpt-3.5-turbo")
        state["conversation_summary"] = summary
        
        # update user preferences
        state = update_user_prference(state, model_name="gpt-3.5-turbo")
        
        return state
    except Exception as e:
        print(f"Error calling LLM model: {e}")
        state["error_message"] = "抱歉，處理您的請求時發生錯誤。"
        return state

            