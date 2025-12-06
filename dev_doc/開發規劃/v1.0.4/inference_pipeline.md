你該準備什麼專案？（按優先順序）
P0 級（必做，面試必問）：推論系統 Portfolio
專案名稱：Production AI Inference Systems - Performance Optimization Portfolio

內容：你已有的兩個專案（詐欺偵測 + 智慧零售）的推論架構深度展示

重點突出：

推論 pipeline 設計（data preprocessing → model inference → post-processing → result serving）

推論效能優化（latency 優化、throughput 提升、資源降低）

推論監控與告警（Prometheus metrics、Grafana dashboard、SLA tracking）

推論成本優化（ONNX 加速、Docker 瘦身、資源調配）

千萬不要：花篇幅講「模型怎麼訓練的」「用了什麼 loss function」「hyperparameter tuning」→ 這些都不是重點！

P1 級（必做，彌補 LLM 缺口）：LLM Inference Benchmarking
專案名稱：LLM Inference Performance Benchmark: TensorRT-LLM vs vLLM vs llama.cpp

為什麼是推論，不是訓練？

系微不在乎「你會不會 fine-tune Llama」

系微在乎「你會不會選擇最快的 inference framework」「你會不會優化 inference latency」

測試項目（全部都是推論）：

text
✅ Inference latency (TTFT, token/sec)
✅ Throughput (concurrent requests handling)
✅ Memory efficiency (GPU memory usage, KV-cache)
✅ Quantization impact (FP16 vs INT8 vs INT4)
✅ Batching strategy (static vs dynamic batching)
✅ Cost analysis (token/dollar, infrastructure sizing)

❌ Training speed
❌ Fine-tuning convergence
❌ Distributed training efficiency
P2 級（加分項）：推論監控 Dashboard
專案名稱：AI Inference Monitoring Dashboard Design

為什麼是推論？

JD Responsibility #3：「Develop tools to measure, visualize, and track system performance」​

這裡的「system」是指「inference serving system」，不是「training cluster」

Dashboard 要監控的指標（全部推論相關）：

text
✅ Inference latency (p50/p95/p99)
✅ Throughput (requests/sec)
✅ GPU utilization (inference workload)
✅ Token usage (LLM serving cost)
✅ Error rate (inference failures)

❌ Training loss curve
❌ Gradient norm
❌ Learning rate schedule
P3 級（補強技術）：PyTorch Profiling
專案類型：還是推論！

實驗設計：

text
✅ Profile YOLO/ResNet **inference** (不是 training)
✅ 分析 inference latency breakdown
✅ 找出 inference bottleneck (哪個 layer 最慢)
✅ 測試不同 batch size 對 inference throughput 的影響

❌ Profile training backward pass
❌ 分析 gradient computation
為什麼？

系微要你優化的是「模型推論」，不是「模型訓練」

PyTorch Profiler 可以用在 training 也可以用在 inference，但你要展現的是「inference profiling」

訓練相關知識要不要準備？
要，但只需要「理解層面」，不需要「實作專案」
準備程度：能在面試時「講清楚訓練 → 推論的關係」就夠了