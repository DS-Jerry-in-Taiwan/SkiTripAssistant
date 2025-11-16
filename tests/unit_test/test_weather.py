import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import pytest
import json
from graph import call_weather_agent

def test_weather_agent_single_day():
    """測試 Weather Agent 查詢單一天氣"""
    location = "台中"
    date = "2025-11-17"
    result = call_weather_agent(location, date)
    print("單一天氣查詢結果：")
    print(result)
    # 檢查回傳內容格式
    assert "天氣預報" in result or "查詢地點" in result
    # 檢查 JSON 格式
    try:
        data = json.loads(result)
        assert "天氣預報" in data
        assert data["查詢地點"] == location
    except Exception:
        pytest.fail("回傳結果不是有效的 JSON 格式")

def test_weather_agent_date_range():
    """測試 Weather Agent 查詢多天氣預報"""
    location = "台中"
    start_date = "2025-11-17"
    end_date = "2025-11-19"
    result = call_weather_agent(location, start_date, end_date)
    print("多天氣查詢結果：")
    print(result)
    # 檢查回傳內容格式
    assert "天氣預報" in result or "查詢地點" in result
    # 檢查 JSON 格式
    try:
        data = json.loads(result)
        assert "天氣預報" in data
        assert data["查詢地點"] == location
        assert data["查詢日期範圍"] == f"{start_date} 至 {end_date}"
        assert isinstance(data["天氣預報"], list)
        assert len(data["天氣預報"]) > 0
    except Exception:
        pytest.fail("回傳結果不是有效的 JSON 格式")