import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import pytest
from graph import create_graph


def test_multi_agent_workflow_basic():
    """
    測試協調器節點與多 Agent 節點串接是否正確，
    包含資料流動、回應整合、流程結束。
    """
    workflow = create_graph()
    # 初始 state，模擬使用者輸入
    state = {
        "user_input": "我想要三天滑雪行程，包含溫泉和美食",
        "user_preferences": {
            "預算": "中等",
            "交通": "大眾運輸",
            "活動": ["滑雪", "溫泉", "美食"]
        }
    }
    # 執行 workflow
    result_state = workflow.invoke(state, config={"configurable": {"thread_id": "test_thread_002"}})
    # 檢查協作流程是否結束
    assert "final_itinerary" in result_state
    assert isinstance(result_state["final_itinerary"], str)
    # 檢查資料流動
    assert "planner_result" in result_state
    assert "conversation_history" in result_state
    # 檢查優化次數不超過 2
    assert result_state.get("optimization_count", 0) <= 2

def test_agent_data_consistency():
    """
    測試 Agent 間資料格式一致性與回傳結構化結果
    """
    workflow = create_graph()
    state = {
        "user_input": "請推薦滑雪課程和溫泉",
        "user_preferences": {}
    }
    result_state = workflow.invoke(state, config={"configurable": {"thread_id": "test_thread_002"}})
    # 檢查 retriever/attraction 結果格式
    if "retriever_result" in result_state:
        assert isinstance(result_state["retriever_result"], str)
    if "attraction_result" in result_state:
        assert isinstance(result_state["attraction_result"], str)
    # 檢查行程格式
    assert "final_itinerary" in result_state

def test_coordinator_entry_point():
    workflow = create_graph()
    state = {"user_input": "test"}
    result_state = workflow.invoke(state, config={"configurable": {"thread_id": "test_thread_003"}})
    # 檢查是否有 planner_result，代表流程有經過 planner 節點
    assert "planner_result" in result_state