# Travel Agent MVP — Inference Pipeline 開發規劃與 Checklist

根據「Production AI Inference Systems - Performance Optimization Portfolio」的面試導向規劃，將本專案的 inference pipeline 分為四大階段，每階段附上開發重點與 checklist，確保專案能完整展現推論系統設計、效能優化、監控與成本分析能力。

---

## P0 級：推論 Pipeline 設計與效能優化

### 目標
- 建立明確的推論流程（資料前處理 → 模型推論 → 後處理 → 結果服務）
- 優化推論效能（latency、throughput、資源利用）

#### 1. 明確劃分 pipeline 各階段
- 定義每個階段的職責（preprocessing, inference, postprocessing, serving）
- 在程式碼中以 class/function/模組區分

#### 2. 將每個階段模組化
- 為每個階段建立獨立的 Python module 或 class
- 設計統一的資料流介面（如 input/output dict 或 dataclass）

#### 3. 實作 API latency 量測
- 在 inference 階段加入 time.perf_counter() 或 time.monotonic() 量測
- 回傳 latency 結果，並記錄於 log 或 metrics

#### 4. 支援 batch inference
- 檢查 OpenAI API 是否支援 batch 請求
- 若支援，設計 batch 資料流與回傳格式
- 若不支援，設計 batch 處理流程（如多執行緒/async 請求）

#### 5. 優化資料流與記憶體使用
- 檢查每階段的資料結構，避免不必要的複製
- 釋放不再使用的變數，減少記憶體佔用
- 如有大檔案/大物件，考慮用生成器（generator）或流式處理

#### 6. 撰寫單元測試覆蓋 pipeline 關鍵路徑
- 為每個階段撰寫 pytest 單元測試
- 測試正常流程、異常輸入、邊界情境

#### 7. 文件化 pipeline 架構與效能優化策略
- 用 markdown/圖示說明 pipeline 架構
- 記錄每個優化點的設計思路與效能數據

### Checklist
- [ ] 明確劃分 pipeline 各階段（preprocessing, inference, postprocessing, serving）
- [ ] 將每個階段模組化，便於維護與擴展
- [ ] 實作 API latency 量測（如 time/perf_counter）
- [ ] 支援 batch inference（如 OpenAI API 支援時）
- [ ] 優化資料流與記憶體使用
- [ ] 撰寫單元測試覆蓋 pipeline 關鍵路徑
- [ ] 文件化 pipeline 架構與效能優化策略

---

## P1 級：LLM Inference Benchmarking

### 目標
- 針對 LLM API（如 OpenAI）進行推論效能基準測試
- 報告 latency、throughput、token/sec、成本等指標

### Checklist
- [ ] 設計並實作推論效能量測腳本（TTFT、token/sec）
- [ ] 支援多種 batch size、並發請求測試
- [ ] 統計與可視化 latency（p50/p95/p99）、throughput
- [ ] 記錄 API token 使用量與推論成本（token/dollar）
- [ ] 撰寫效能報告與分析（含圖表）
- [ ] 文件化 benchmarking 方法與結果

---

## P2 級：推論監控與告警

### 目標
- 建立推論系統監控（Prometheus/Grafana）
- 追蹤 SLA、效能異常自動告警

### Checklist
- [ ] 導入 Prometheus metrics（latency, throughput, error rate, token usage）
- [ ] 設計 Grafana dashboard 展示關鍵指標
- [ ] 設定 SLA 閾值與自動告警（如 latency 超標、error rate 過高）
- [ ] 撰寫監控系統部署與維護文件
- [ ] 定期回顧與優化監控指標

---

## P3 級：推論成本與資源優化

### 目標
- 分析推論成本結構，提出優化建議
- 優化 Docker 映像、API 請求策略、資源配置

### Checklist
- [ ] 統計推論 token 使用量與 API 成本
- [ ] 分析不同 batch/並發策略對成本的影響
- [ ] 優化 Dockerfile，減少映像體積
- [ ] 文件化成本分析與優化建議
- [ ] 準備面試用的成本優化案例說明

---

## 補充：面試展示建議

- 準備一份簡明的 README/報告，說明 pipeline 架構、效能數據、監控設計與成本分析
- 強調「推論」而非「訓練」的技術深度
- 展示效能優化、監控、成本分析的具體成果（如 dashboard 截圖、效能報告）

---

> 依照上述 checklist 逐步開發與驗證，能讓你的專案完整對齊 AI Inference Engineer 的面試需求。