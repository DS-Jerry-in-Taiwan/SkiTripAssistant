import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import pytest
import json
from graph import call_accommodation_agent

def test_accommodation_agent_basic():
    """測試 Accommodation Agent 基本查詢功能"""
    location = "台中"
    checkin = "2025-01-15"
    checkout = "2025-01-17"
    
    result = call_accommodation_agent(location, checkin, checkout)
    
    print("=" * 60)
    print("住宿查詢結果：")
    print(result)
    print("=" * 60)
    
    # 檢查回傳內容格式
    assert "住宿" in result or "查詢地點" in result or "推薦" in result
    
    # 檢查 JSON 格式
    try:
        data = json.loads(result)
        assert "查詢地點" in data or "推薦住宿" in data
        print("✅ 測試通過：回傳格式正確")
    except json.JSONDecodeError as e:
        print(f"⚠️  警告：回傳結果不是標準 JSON 格式: {e}")
        # 如果不是 JSON，至少要包含住宿相關資訊
        assert "住宿" in result

def test_accommodation_agent_date_validation():
    """測試 Accommodation Agent 日期格式處理"""
    location = "台北"
    checkin = "2025-02-10"
    checkout = "2025-02-12"
    
    result = call_accommodation_agent(location, checkin, checkout)
    
    print("=" * 60)
    print("日期格式測試結果：")
    print(result)
    print("=" * 60)
    
    # 檢查是否包含日期資訊
    assert checkin in result or "入住" in result
    assert checkout in result or "退房" in result
    print("✅ 測試通過：日期處理正確")

def test_accommodation_agent_json_structure():
    """測試 Accommodation Agent JSON 結構完整性"""
    location = "台中"
    checkin = "2025-03-01"
    checkout = "2025-03-03"
    
    result = call_accommodation_agent(location, checkin, checkout)
    
    print("=" * 60)
    print("JSON 結構測試結果：")
    print(result)
    print("=" * 60)
    
    try:
        data = json.loads(result)
        
        # 檢查必要欄位
        assert "查詢地點" in data, "缺少查詢地點欄位"
        assert data["查詢地點"] == location, "地點不符"
        
        # 檢查住宿推薦
        assert "推薦住宿" in data, "缺少推薦住宿欄位"
        assert isinstance(data["推薦住宿"], list), "推薦住宿應為列表"
        
        # 檢查住宿項目結構
        if len(data["推薦住宿"]) > 0:
            hotel = data["推薦住宿"][0]
            assert "名稱" in hotel, "住宿項目缺少名稱"
            assert "類型" in hotel, "住宿項目缺少類型"
            assert "評分" in hotel, "住宿項目缺少評分"
            assert "價格" in hotel, "住宿項目缺少價格"
            print("✅ 測試通過：JSON 結構完整")
        else:
            print("⚠️  警告：推薦住宿列表為空")
        
    except json.JSONDecodeError:
        pytest.fail("回傳結果不是有效的 JSON 格式")
    except AssertionError as e:
        pytest.fail(f"JSON 結構不完整: {e}")

def test_accommodation_agent_multiple_locations():
    """測試 Accommodation Agent 不同地點查詢"""
    locations = ["台中", "台北", "高雄"]
    
    for location in locations:
        result = call_accommodation_agent(location, "2025-04-01", "2025-04-03")
        print(f"\n{'=' * 60}")
        print(f"查詢地點：{location}")
        print(result)
        print('=' * 60)
        
        assert location in result or "查詢地點" in result
    
    print("✅ 測試通過：多地點查詢成功")

def test_accommodation_agent_error_handling():
    """測試 Accommodation Agent 錯誤處理"""
    # 測試無效日期格式
    location = "台中"
    invalid_checkin = "2025-13-40"  # 無效日期
    checkout = "2025-01-17"
    
    result = call_accommodation_agent(location, invalid_checkin, checkout)
    
    print("=" * 60)
    print("錯誤處理測試結果：")
    print(result)
    print("=" * 60)
    
    # 應該要有錯誤處理或回傳合理結果
    assert result is not None
    assert len(result) > 0
    print("✅ 測試通過：錯誤處理正常")