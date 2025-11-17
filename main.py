import uuid
import traceback
from dotenv import load_dotenv
from graph import create_graph
from state import AgentState

def display_welcome():
    print("=" * 40)
    print("🤖 歡迎使用旅遊推薦代理人！請輸入您的旅遊需求，或輸入 'exit' 離開程式。")
    print("=" * 10)
    print("請輸入您的旅遊問題，或輸入 exit 離開")
    print("提示：您可以輸入像是 '推薦台北景點'、'我想去日本旅遊' 等需求。")
    print("=" * 40)
def main():
    load_dotenv()
    display_welcome()
    app = create_graph()
    thread_id = str(uuid.uuid4())
    state = {
        "messages": [],
        "user_input": "",
        "planner_result": "",
        "recommendation_result": "",
        "evaluation_result": "",
        "current_agent": "",
        "need_evaluation": False
    }
    print("🤖 歡迎使用旅遊推薦代理人！請輸入您的旅遊需求，或輸入 'exit' 離開程式。")
    while True:
        user_input = input("👤 使用者：")
        if user_input.lower() == 'exit':
            print("👋 感謝使用旅遊推薦代理人！祝您旅途愉快！")
            break
        state["messages"].append({"role": "user", "content": user_input})
        all_user_inputs = " ".join([msg["content"] for msg in state["messages"] if msg["role"] == "user"])
        state["query"] = all_user_inputs
        try:
            result_state = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
            state = result_state
        except TypeError as e:
            if "checkin" in str(e) or "checkout" in str(e):
                print("⚠️ 查詢住宿時請提供入住與退房日期（格式 YYYY-MM-DD）")
                continue
            else:
                print(f"⚠️ 執行錯誤：{e}")
                continue

        current_agent = result_state.get("current_agent", "")

        # 顯示推薦/引導階段結果
        if current_agent == "recommendation":
            recommendation_result = result_state.get("recommendation_result", "")
            if recommendation_result:
                print(f"🤖 {recommendation_result}")
                if ("請問" in recommendation_result or "還缺少" in recommendation_result or "資訊不足" in recommendation_result):
                    continue  # 等待使用者補充
            else:
                print("\n🌟 未產生推薦/引導結果。")

        # 顯示行程規劃結果
        if current_agent == "planner":
            planner_result = result_state.get("planner_result", "")
            if planner_result:
                print("\n🗺️ 行程規劃結果：")
                print(planner_result)
            else:
                print("\n🗺️ 未產生行程規劃結果。")

        # 顯示評估結果
        if current_agent == "evaluator":
            evaluation_result = result_state.get("evaluation_result", "")
            if evaluation_result:
                print("\n📊 行程評估結果：")
                print(evaluation_result)
            else:
                print("\n📊 未進行行程評估。")

        # 結束提示
        if current_agent not in ["recommendation", "planner", "evaluator"]:
            print("\n✅ 流程已結束，感謝您的使用！")
            break
if __name__ == "__main__":
    main()