import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


import pytest
import json
from graph import call_evaluator_agent

class TestEvaluatorAgent:
    """Evaluator Agent 單元測試"""
    
    # def test_evaluator_basic_evaluation(self):
    #     """測試 Evaluator Agent 基本評估功能"""
    #     # 模擬行程資料
    #     itinerary_data = {
    #         "summary": "3天2夜台中滑雪溫泉之旅",
    #         "total_budget": "NT$ 15,000",
    #         "transport_plan": "高鐵往返 + 當地接駁",
    #         "days": 3,
    #         "budget_level": "中等",
    #         "daily_plans": [
    #             {
    #                 "date": "Day 1",
    #                 "activities": [
    #                     {
    #                         "time": "09:00-12:00",
    #                         "activity": "滑雪課程",
    #                         "location": "雪山滑雪場",
    #                         "transport": "高鐵+接駁車",
    #                         "notes": "建議預約初學者課程"
    #                     },
    #                     {
    #                         "time": "14:00-17:00",
    #                         "activity": "溫泉體驗",
    #                         "location": "台中溫泉會館",
    #                         "transport": "接駁車",
    #                         "notes": "享受露天溫泉"
    #                     }
    #                 ],
    #                 "meals": {
    #                     "breakfast": "飯店早餐",
    #                     "lunch": "滑雪場餐廳",
    #                     "dinner": "台中市區美食"
    #                 },
    #                 "accommodation": "台中溫泉會館"
    #             },
    #             {
    #                 "date": "Day 2",
    #                 "activities": [
    #                     {
    #                         "time": "09:00-11:00",
    #                         "activity": "宮原眼科",
    #                         "location": "台中市區",
    #                         "transport": "計程車",
    #                         "notes": "品嚐冰淇淋"
    #                     },
    #                     {
    #                         "time": "11:30-13:00",
    #                         "activity": "台中國家歌劇院",
    #                         "location": "台中市區",
    #                         "transport": "步行",
    #                         "notes": "建築參觀"
    #                     }
    #                 ],
    #                 "meals": {
    #                     "breakfast": "飯店早餐",
    #                     "lunch": "歌劇院餐廳",
    #                     "dinner": "逢甲夜市"
    #                 },
    #                 "accommodation": "台中市區飯店"
    #             }
    #         ]
    #     }
        
    #     # 使用者偏好
    #     user_preferences = {
    #         "budget_level": "中等",
    #         "days": 3,
    #         "interests": ["滑雪", "溫泉", "美食", "文化"]
    #     }
        
    #     result = call_evaluator_agent(itinerary_data, user_preferences)
        
    #     print("\n" + "=" * 60)
    #     print("評估結果：")
    #     print(result)
    #     print("=" * 60)
        
    #     # 檢查 JSON 格式
    #     try:
    #         data = json.loads(result)
            
    #         # 檢查必要欄位
    #         assert "行程摘要" in data, "缺少行程摘要欄位"
    #         assert "評分" in data, "缺少評分欄位"
    #         assert "優化建議" in data, "缺少優化建議欄位"
    #         assert "整體評價" in data, "缺少整體評價欄位"
    #         assert "是否需要調整" in data, "缺少是否需要調整欄位"
            
    #         # 檢查評分結構
    #         scores = data["評分"]
    #         assert "預算合理性" in scores, "缺少預算合理性評分"
    #         assert "時間安排" in scores, "缺少時間安排評分"
    #         assert "交通便利性" in scores, "缺少交通便利性評分"
    #         assert "活動豐富度" in scores, "缺少活動豐富度評分"
    #         assert "整體評分" in scores, "缺少整體評分"
            
    #         # 檢查優化建議結構
    #         suggestions = data["優化建議"]
    #         assert isinstance(suggestions, list), "優化建議應為列表"
            
    #         if len(suggestions) > 0:
    #             suggestion = suggestions[0]
    #             assert "類型" in suggestion, "優化建議缺少類型"
    #             assert "原因" in suggestion, "優化建議缺少原因"
    #             assert "建議" in suggestion, "優化建議缺少建議內容"
    #             assert "優先級" in suggestion, "優化建議缺少優先級"
            
    #         print("✅ 測試通過：評估結果格式正確")
            
    #     except json.JSONDecodeError as e:
    #         pytest.fail(f"回傳結果不是有效的 JSON 格式: {e}")
    #     except AssertionError as e:
    #         pytest.fail(f"JSON 結構不完整: {e}")
    
    # def test_evaluator_score_range(self):
    #     """測試評分範圍是否合理（0-10）"""
    #     itinerary_data = {
    #         "summary": "2天1夜台北快閃之旅",
    #         "total_budget": "NT$ 8,000",
    #         "transport_plan": "高鐵往返",
    #         "days": 2,
    #         "budget_level": "經濟",
    #         "daily_plans": [
    #             {
    #                 "date": "Day 1",
    #                 "activities": [
    #                     {
    #                         "time": "10:00-12:00",
    #                         "activity": "台北101",
    #                         "location": "信義區",
    #                         "transport": "捷運"
    #                     }
    #                 ],
    #                 "meals": {
    #                     "breakfast": "便利商店",
    #                     "lunch": "美食街",
    #                     "dinner": "夜市"
    #                 },
    #                 "accommodation": "青年旅館"
    #             }
    #         ]
    #     }
        
    #     user_preferences = {
    #         "budget_level": "經濟",
    #         "days": 2,
    #         "interests": ["觀光", "美食"]
    #     }
        
    #     result = call_evaluator_agent(itinerary_data, user_preferences)
    #     data = json.loads(result)
        
    #     print("\n" + "=" * 60)
    #     print("評分範圍測試：")
        
    #     scores = data["評分"]
    #     for key, value in scores.items():
    #         print(f"{key}: {value}")
    #         assert 0 <= value <= 10, f"{key} 評分超出範圍 (0-10): {value}"
        
    #     print("=" * 60)
    #     print("✅ 測試通過：評分範圍正確 (0-10)")
    
    # def test_evaluator_optimization_suggestions(self):
    #     """測試優化建議的完整性"""
    #     # 故意設計一個有問題的行程
    #     itinerary_data = {
    #         "summary": "1天極限挑戰",
    #         "total_budget": "NT$ 50,000",  # 預算過高
    #         "transport_plan": "自駕",
    #         "days": 1,
    #         "budget_level": "豪華",
    #         "daily_plans": [
    #             {
    #                 "date": "Day 1",
    #                 "activities": [
    #                     {
    #                         "time": "08:00-09:00",
    #                         "activity": "台北101",
    #                         "location": "台北",
    #                         "transport": "自駕"
    #                     },
    #                     {
    #                         "time": "09:30-10:30",
    #                         "activity": "日月潭",
    #                         "location": "南投",
    #                         "transport": "自駕 (3小時車程)"  # 時間不合理
    #                     },
    #                     {
    #                         "time": "11:00-12:00",
    #                         "activity": "阿里山",
    #                         "location": "嘉義",
    #                         "transport": "自駕 (2.5小時車程)"  # 時間不合理
    #                     }
    #                 ],
    #                 "meals": {
    #                     "breakfast": "五星飯店",
    #                     "lunch": "米其林餐廳",
    #                     "dinner": "高級日料"
    #                 },
    #                 "accommodation": "無"
    #             }
    #         ]
    #     }
        
    #     user_preferences = {
    #         "budget_level": "中等",  # 與行程預算不符
    #         "days": 1,
    #         "interests": ["觀光"]
    #     }
        
    #     result = call_evaluator_agent(itinerary_data, user_preferences)
    #     data = json.loads(result)
        
    #     print("\n" + "=" * 60)
    #     print("優化建議測試（預期有多個建議）：")
    #     print(json.dumps(data["優化建議"], ensure_ascii=False, indent=2))
    #     print("=" * 60)
        
    #     # 應該有優化建議
    #     assert len(data["優化建議"]) > 0, "應該有優化建議"
        
    #     # 應該需要調整
    #     assert data["是否需要調整"] == True, "應該需要調整行程"
        
    #     # 檢查優先級
    #     priorities = [s["優先級"] for s in data["優化建議"]]
    #     assert any(p in ["高", "中", "低"] for p in priorities), "優先級格式不正確"
        
    #     print("✅ 測試通過：優化建議生成正確")
    
    # def test_evaluator_perfect_itinerary(self):
    #     """測試完美行程（應不需要調整）"""
    #     itinerary_data = {
    #         "summary": "3天2夜台中悠閒之旅",
    #         "total_budget": "NT$ 12,000",
    #         "transport_plan": "高鐵往返 + 公車",
    #         "days": 3,
    #         "budget_level": "中等",
    #         "daily_plans": [
    #             {
    #                 "date": "Day 1",
    #                 "activities": [
    #                     {
    #                         "time": "10:00-12:00",
    #                         "activity": "宮原眼科",
    #                         "location": "台中市區",
    #                         "transport": "公車",
    #                         "notes": "留充足時間"
    #                     },
    #                     {
    #                         "time": "14:00-17:00",
    #                         "activity": "台中國家歌劇院",
    #                         "location": "台中市區",
    #                         "transport": "公車",
    #                         "notes": "輕鬆遊覽"
    #                     }
    #                 ],
    #                 "meals": {
    #                     "breakfast": "飯店早餐",
    #                     "lunch": "市區餐廳",
    #                     "dinner": "逢甲夜市"
    #                 },
    #                 "accommodation": "台中市區商旅"
    #             }
    #         ]
    #     }
        
    #     user_preferences = {
    #         "budget_level": "中等",
    #         "days": 3,
    #         "interests": ["美食", "文化", "輕鬆"]
    #     }
        
    #     result = call_evaluator_agent(itinerary_data, user_preferences)
    #     data = json.loads(result)
        
    #     print("\n" + "=" * 60)
    #     print("完美行程測試：")
    #     print(f"整體評分: {data['評分']['整體評分']}")
    #     print(f"是否需要調整: {data['是否需要調整']}")
    #     print("=" * 60)
        
    #     # 評分應該較高
    #     assert data["評分"]["整體評分"] >= 7.0, "完美行程整體評分應 >= 7.0"
        
    #     print("✅ 測試通過：完美行程評估正確")
    
    def test_evaluator_error_handling(self):
        """測試錯誤處理（空行程資料）"""
        itinerary_data = {}
        user_preferences = {"budget_level": "中等", "days": 3}
        
        result = call_evaluator_agent(itinerary_data, user_preferences)
        
        print("\n" + "=" * 60)
        print("錯誤處理測試（空行程）：")
        print(result)
        print("=" * 60)
        
        # 應該要有回應（不應該崩潰）
        assert result is not None
        assert len(result) > 0
        
        # 嘗試解析 JSON
        try:
            data = json.loads(result)
            assert "評分" in data or "error" in data
            print("✅ 測試通過：錯誤處理正常")
        except:
            # 如果不是 JSON，至少要有內容
            assert "評估" in result or "錯誤" in result or "error" in result
            print("✅ 測試通過：錯誤處理正常（非 JSON 回應）")

# # ========== 執行測試 ==========
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])