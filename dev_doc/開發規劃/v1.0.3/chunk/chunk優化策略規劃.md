### Chunking 策略優化規劃

**目標**  
提升旅遊知識文件分割的語意連貫性與檢索準確度，避免重要資訊被切斷。

---

#### 1. 固定長度 + Overlap 策略
- chunk_size 設定為 300~600 字（或 tokens）
- chunk_overlap 設定為 50~100 字
- 適用於一般旅遊文件，確保景點、路線等資訊完整

## 固定長度 + Overlap 策略開發步驟

1. 分析旅遊知識文件內容型態，確認適合 chunk_size（建議 300~600 字或 tokens）。
2. 在 `build_vectorstore.py` 中，使用 LangChain 的 `RecursiveCharacterTextSplitter`。
3. 設定 chunk_size 為 300~600，chunk_overlap 為 50~100。
4. 將旅遊知識文件分割為 chunk，並存入 Chroma 向量資料庫。
5. 編寫單元測試，驗證分割後的 chunk 是否語意連貫、資訊完整。
6. 測試檢索效果，根據查詢回應品質微調 chunking 參數。


#### 2. 語義分割策略
- 根據段落、標題、主題分割（如 MarkdownHeaderTextSplitter）
- 適用於結構化文件，讓每個 chunk 保持語意完整

## 語義分割策略開發步驟

1. 分析旅遊知識文件結構，確認是否有明確段落、標題或主題。
2. 在 `build_vectorstore.py` 中，導入並使用 LangChain 的 `MarkdownHeaderTextSplitter` 或其他語義分割工具。
3. 設定分割規則（如根據 Markdown 標題、段落、主題等）。
4. 將結構化分割後的 chunk 存入 Chroma 向量資料庫。
5. 編寫單元測試，驗證每個 chunk 是否語意完整、主題明確。
6. 測試檢索效果，根據查詢回應品質微調分割規則與參數。
7. 收集用戶回饋，持續優化語義分割策略。

### 主流語意切分方法
    ** Markdown 標題分割 **

    依據 Markdown 標題（如 #, ##, ###）分割文本，常用於結構化文件。
    工具：MarkdownHeaderTextSplitter（LangChain）
    
    ** 固定長度 + Overlap 分割 **

    依字數或 token 數分割，並設定重疊區，確保語意連貫。
    工具：RecursiveCharacterTextSplitter（LangChain）
    
    ** 正則表達式分割 **

    針對特殊符號（如「【主題】」）自訂分割規則，保留主題語意。
    適用於非標準格式文本。
    
    ** 語義分割（Semantic Splitter） **

    利用 NLP 工具（如 spaCy、NLTK）根據語句、段落、主題自動分割。
    適合自然語言長文本。

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

**Checklist（優化開發全流程）**
- [x] 分析旅遊知識文件內容型態
- [x] 選擇合適的 chunking 策略（固定長度+overlap 或語義分割）
- [x] 實作並測試分割工具（LangChain TextSplitter）
- [x] 驗證分割後的檢索品質
- [ ] 根據用戶查詢情境動態調整查詢參數（如 k 值、分數閾值）
- [ ] 支援多輪查詢聚合（recent_queries 合併）
- [ ] 測試查詢階段動態選取/聚合 chunk 的效果
- [ ] 收集用戶回饋，持續微調 chunking 與查詢策略