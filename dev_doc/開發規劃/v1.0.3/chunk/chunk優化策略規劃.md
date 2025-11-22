### Chunking 策略優化規劃

**目標**  
提升旅遊知識文件分割的語意連貫性與檢索準確度，避免重要資訊被切斷。

---

#### 1. 固定長度 + Overlap 策略
- chunk_size 設定為 300~600 字（或 tokens）
- chunk_overlap 設定為 50~100 字
- 適用於一般旅遊文件，確保景點、路線等資訊完整

#### 2. 語義分割策略
- 根據段落、標題、主題分割（如 MarkdownHeaderTextSplitter）
- 適用於結構化文件，讓每個 chunk 保持語意完整

#### 3. 滑動窗口分割
- 使用 RecursiveCharacterTextSplitter，支援 chunk_overlap
- 適合長文本或資訊密集型內容

#### 4. 多輪查詢/主題聚合
- 根據最近幾輪 user query 合併 chunk，提升主題聚合度
- 動態調整 chunk_size，根據查詢複雜度自動分割

---

**落地技術建議**
- 使用 LangChain 的 `RecursiveCharacterTextSplitter` 或 `MarkdownHeaderTextSplitter` 實作分割
- 在 `build_vectorstore.py` 中調整 chunking 參數
- 測試不同策略下的檢索效果，選擇最適合旅遊知識文件的分割方式

---

**Checklist**
- [ ] 分析旅遊知識文件內容型態
- [ ] 選擇合適的 chunking 策略（固定長度+overlap 或語義分割）
- [ ] 實作並測試分割工具（LangChain TextSplitter）
- [ ] 驗證分割後的檢索品質
- [ ] 根據用戶查詢情境動態調整 chunking 參數