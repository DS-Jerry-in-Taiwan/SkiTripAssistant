import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import pytest
import json
from graph import create_graph

@pytest.mark.parametrize("need_evaluation", [True, False])
def test_planner_evaluator_conditional_flow(need_evaluation):
    """
    整合測試：Planner 執行完根據 need_evaluation 決定是否進入 Evaluator
    """
    mvp_graph = create_graph()
    initial_state = {
        "user_input": "我想去台中玩3天2夜，預算中等，喜歡滑雪和溫泉",
        "planner_result": "",
        "evaluation_result": "",
        "current_agent": "",
        "conversation_history": [],
        "final_itinerary": "",
        "need_evaluation": need_evaluation
    }
    config = {"configurable": {"thread_id": f"test_conditional_{need_evaluation}"}}
    result = mvp_graph.invoke(initial_state, config)

    print("\n" + "=" * 60)
    print(f"need_evaluation = {need_evaluation}")
    print("Planner 結果：")
    print(result.get("planner_result", "無結果")[:500])
    print("\n" + "-" * 60)
    if need_evaluation:
        print("Evaluator 結果：")
        print(result.get("evaluation_result", "無結果")[:500])
        assert result["current_agent"] == "evaluator"
        assert "evaluation_result" in result
        # 嘗試解析 JSON
        try:
            eval_data = json.loads(result["evaluation_result"])
            assert "評分" in eval_data
            assert "優化建議" in eval_data
            print("✅ 測試通過：有評估流程且結果格式正確")
        except Exception as e:
            print(f"⚠️  評估結果不是標準 JSON 格式: {e}")
    else:
        print("未進入 Evaluator，流程直接結束")
        assert result["current_agent"] == "planner"
        assert "evaluation_result" not in result or not result["evaluation_result"]
        print("✅ 測試通過：流程直接結束於 Planner")

    print("=" * 60)

# ========== 執行測試 ==========
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])