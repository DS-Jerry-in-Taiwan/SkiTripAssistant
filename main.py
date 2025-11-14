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
    state = AgentState(messages=[], user_preferences={}, retrieved_docs=[], query="")
    while True:
        user_input = input("👤 使用者：")
        if user_input.lower() == 'exit':
            print("👋 感謝使用旅遊推薦代理人！祝您旅途愉快！")
            break
        state["messages"].append({"role": "user", "content": user_input})
        state["query"] = user_input
        print("🤖 代理人正在處理您的需求，請稍候...")
        retry_count = 0
        while retry_count < 3:
            try:
                result_state = app.invoke(input=state, config={"configurable": {"thread_id": thread_id}})
                ai_msg = [msg for msg in result_state["messages"] if msg["role"] == "assistant"]
                if ai_msg:
                    print(f"🤖 代理人：{ai_msg[-1]['content']}")
                else:
                    print("🤖 代理人：抱歉，未能生成回應。")
                state = result_state
                break
            except Exception as e:
                print(f"❌ 發生錯誤：{e}")
                traceback.print_exc()
                retry_count += 1
                if retry_count < 3:
                    print(f"🤖 正在重試...（第 {retry_count} 次）")
                else:
                    print("❌ 多次嘗試後仍無法處理您的需求，請稍後再試。")
                    continue
                    
            
if __name__ == "__main__":
    main()
        