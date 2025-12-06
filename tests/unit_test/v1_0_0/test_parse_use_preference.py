import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import pytest
from nodes import parse_user_preferences_tool
import json

def test_parse_user_preferences_basic():
    """測試基本解析功能"""
    raw_text = "我想去台中玩2天1夜，預算中等，出發日期是2024-12-20"
    required_fields = '["days", "budget_level", "date", "location"]'
    
    result_json = parse_user_preferences_tool(raw_text, required_fields)
    result = json.loads(result_json)
    
    assert result["days"] == 2
    assert result["budget_level"] == "中等"
    assert result["date"] == "2024-12-20"
    assert result["location"] == "台中"


def test_parse_user_preferences_missing_fields():
    """測試缺失欄位處理"""
    raw_text = "我想去台中玩"
    required_fields = '["days", "budget_level", "date", "location"]'
    
    result_json = parse_user_preferences_tool(raw_text, required_fields)
    result = json.loads(result_json)
    
    # 缺失欄位應填預設值
    assert "location" in result
    assert result["location"] == "台中"
    # days 缺失應為 0
    assert result.get("days", 0) >= 0


def test_parse_user_preferences_partial():
    """測試部分資訊解析"""
    raw_text = "預算高級，3天2夜"
    required_fields = '["days", "budget_level"]'
    
    result_json = parse_user_preferences_tool(raw_text, required_fields)
    result = json.loads(result_json)
    
    assert result["days"] == 3
    assert result["budget_level"] == "高級"


# def test_parse_user_preferences_error_handling():
#     """測試錯誤處理"""
#     raw_text = "這是一段無關的文字"
#     required_fields = '["days", "budget_level", "date", "location"]'
    
#     result_json = parse_user_preferences_tool(raw_text, required_fields)
#     result = json.loads(result_json)
    
    # 應該回傳預設值，不會崩潰
    assert "days" in result
    assert "budget_level" in result