from prometheus_client import Histogram, start_http_server
import time 
import random

# 定義一個 Histogram 來記錄延遲時間
AGENT_LATENCY = Histogram('agent_latency_seconds', 'Latency of agent invocations', ['agent_name'])

def fake_inference(agent_name):
    """模擬 agent 推理過程，隨機延遲 0.5 到 2 秒"""
    start = time.perf_counter()
    time.sleep(random.uniform(0.2, 1.2)) # 模擬推理時間
    latency = time.perf_counter() - start
    AGENT_LATENCY.labels(agent_name=agent_name).observe(latency)
    print(f"{agent_name} completed in {latency:.2f} seconds")
    return latency

if __name__ == "__main__":
    # 啟動prometheus metrics server
    start_http_server(8000)
    print("Prometheus metrics server started on port 8000")
    while True:
        # 模擬不同 agent 的推理
        fake_inference("planner_node")
        fake_inference("evaluator_node")
        fake_inference("recommendation_node")
        time.sleep(5) # 每隔5秒模擬一次