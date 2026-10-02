# DocuClarify 🔍

> **Local-First 文件智能問答與精確原文引用系統**  
> 混合檢索（BM25 + 向量 RRF）・連續段落拼接・Markdown 排版・純本地向量儲存・零金鑰代管（Pure BYOK）・Docker 一鍵啟動

---

## 🌟 核心特色

1. **混合檢索（Hybrid Search：BM25 關鍵字 ＋ 向量語意 ＋ RRF 融合）**：
   - 內建輕量 BM25Okapi 稀疏檢索（支援英文單字與中日韓雙字元 n-gram 斷詞）與 ChromaDB 向量檢索。
   - 透過 **Reciprocal Rank Fusion (RRF)** 演算法融合兩者排名，兼顧專有名詞、法規代號、精確數字與抽象概念的檢索準確度。

2. **打字機即時串流輸出（SSE Streaming Response）**：
   - 採用 Server-Sent Events (SSE) 協定，檢索命中後立即推送引用來源卡片，隨後將生成文字逐字即時串流（毫秒級延遲），大幅提升體感流暢度。

3. **完整 Markdown 渲染與排版**：
   - 整合 Marked.js，完美支援清單、表格、粗體、引言及程式碼區塊的高質感排版。
   - 提供每個回答一鍵「複製內容」與頂部「匯出 Markdown 對話紀錄」功能。

4. **相鄰連續段落無縫拼接（Adjacent Chunk Merging & Auto-Stitching）**：
   - 自動偵測同一文件中連續序號的段落區塊，運用後綴-前綴重疊比對演算法剪除 Overlap 重複字元，拼合成完整長段落，徹底解決「跨邊界斷句與斷章取義」問題。

5. **多輪對話歷史上下文記憶（Multi-Turn Chat Context）**：
   - 支援基於對話脈絡的深入追問，並提供一鍵清空對話重置按鈕。

6. **零金鑰代管（Zero-Key Dependency / Pure BYOK）**：
   - 專案程式碼與伺服器端不留存任何預設 API Key。
   - 所有 Base URL、Token 與模型名稱均由使用者在瀏覽器介面自行填寫，儲存於使用者本機 `localStorage`。

7. **本機向量儲存與模型隔離（Persistent ChromaDB）**：
   - 採用 ChromaDB 內嵌持久化模式（In-Process Persistent Client），資料自動保存在本機掛載目錄（`./data/chroma`）。
   - Collection 依 Embedding 模型名稱隔離，切換不同維度模型（如 OpenAI 1536 維與 Ollama 768 維）不會發生衝突。

8. **單一容器開箱即用（Single Docker Container）**：
   - 後端 FastAPI 整合非同步 API 與靜態單頁 Web UI，透過一個 `docker-compose.yml` 即可在秒級內啟動。

---

## 🚀 快速啟動（Docker）

### 1. 啟動服務

進入專案目錄，執行以下指令：

```bash
cd DocuClarify
docker compose up -d --build
```

### 2. 開啟瀏覽器

打開瀏覽器進入：
👉 **[http://localhost:8000](http://localhost:8000)**

### 3. 設定 AI 連線（BYOK）

初次使用時，點擊右上角 **「AI 連線設定」**（介面已內建快速預設選單）：
- **Google Gemini 官方 (極力推薦：有充裕免費額度、速度飛快)**：
  - Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/`
  - API Key: `AIzaSy...`（在 Google AI Studio 即可一鍵免費獲取）
  - Chat Model: `gemini-3.8-flash`（或 `gemini-3.5-flash-lite`）
  - Embedding Model: `gemini-embedding-001`（或 `gemini-embedding-2-preview`）
- **Anthropic Claude 官方**：
  - Base URL: `https://api.anthropic.com/v1`
  - API Key: `sk-ant-...`
  - Chat Model: `claude-3-5-sonnet-20241022`（或 `claude-3-5-haiku-20241022`）
  - Embedding Model: `text-embedding-3-small` 或 `gemini-embedding-001`（Anthropic 原生無向量模型，可搭配 OpenAI / Gemini / 本地 Ollama）
- **OpenRouter (Claude 網關)**：
  - Base URL: `https://openrouter.ai/api/v1`
  - API Key: `sk-or-v1-...`
  - Chat Model: `anthropic/claude-3.5-sonnet`
  - Embedding Model: `text-embedding-3-small`
- **OpenAI 官方**：
  - Base URL: `https://api.openai.com/v1`
  - API Key: `sk-...`
  - Chat Model: `gpt-4o-mini`
  - Embedding Model: `text-embedding-3-small`
- **本機 Ollama**（若 Ollama 運行在 Mac 本機，Docker 容器請使用 `host.docker.internal` 連線）：
  - Base URL: `http://host.docker.internal:11434/v1`
  - API Key: （留空即可）
  - Chat Model: `llama3.2`
  - Embedding Model: `nomic-embed-text`
- **DeepSeek**：
  - Base URL: `https://api.deepseek.com/v1`
  - API Key: `sk-...`
  - Chat Model: `deepseek-chat`
  - Embedding Model: `text-embedding-3-small`
- **Groq**：
  - Base URL: `https://api.groq.com/openai/v1`
  - API Key: `gsk-...`
  - Chat Model: `llama-3.3-70b-versatile`
  - Embedding Model: `nomic-embed-text`

點擊 **「測試連線」** 確認無誤後，按下 **「儲存設定」**。

---

## 📂 專案架構

```text
DocuClarify/
├── docker-compose.yml       # Docker Compose 服務編排
├── Dockerfile               # 輕量 Python 3.11-slim 映像檔構建
├── requirements.txt         # 核心依賴（FastAPI, ChromaDB, PyPDF 等）
├── .dockerignore            # 忽略構建緩存與非必要檔案
├── README.md                # 專案詳細說明與架構文件
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 核心服務、SSE 串流與 RAG 路由
│   ├── bm25.py              # 輕量 BM25Okapi 檢索器與 RRF 排名融合
│   ├── ai_client.py         # OpenAI 相容 API 用戶端（連線池、Embedding、Chat 串流）
│   ├── document_parser.py   # PDF / TXT / Markdown 檔案解析、頁碼分塊與無縫拼接
│   ├── vector_store.py      # ChromaDB 向量庫、模型隔離與混合檢索
│   └── static/
│       └── index.html       # 單頁 Web UI（Marked.js 排版 + 串流打字器 + 匯出）
└── data/                    # 掛載目錄（保存 ChromaDB 向量索引與檔案紀錄）
```

---

## 常用指令

- **查看日誌**：
  ```bash
  docker compose logs -f
  ```
- **停止服務**：
  ```bash
  docker compose down
  ```
- **重新構建並啟動**：
  ```bash
  docker compose up -d --build
  ```
