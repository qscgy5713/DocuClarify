from contextlib import asynccontextmanager
import json
import os
import uuid
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ai_client import AIClient, close_async_client
from app.document_parser import DocumentParser
from app.vector_store import VectorStore

# Maximum file size permitted for upload (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_async_client()


app = FastAPI(
    title="DocuClarify API",
    description="Local-First Document Q&A with Bring-Your-Own-Key (BYOK) architecture.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store
vector_store = VectorStore()

# Path to static frontend files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "DocuClarify API is running. UI not found."}


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "app": "DocuClarify", "version": "1.1.0"}


class ConnectionTestRequest(BaseModel):
    base_url: str = Field(..., description="OpenAI-compatible Base URL")
    api_key: Optional[str] = Field(None, description="API Key or Token")
    chat_model: str = Field("gpt-4o-mini", description="Model name to test")


@app.post("/api/test-connection")
async def test_connection(req: ConnectionTestRequest):
    result = await AIClient.test_connection(
        base_url=req.base_url,
        api_key=req.api_key,
        chat_model=req.chat_model,
    )
    return result


@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    base_url: str = Form(...),
    api_key: Optional[str] = Form(None),
    embedding_model: str = Form(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="請提供有效的檔案名稱。")

    clean_filename = os.path.basename(file.filename.strip())

    allowed_exts = (".pdf", ".txt", ".md")
    if not clean_filename.lower().endswith(allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"不支援的檔案格式。目前僅支援：{', '.join(allowed_exts)}",
        )

    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="上傳的檔案內容為空。")

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"檔案大小（{file_size / (1024 * 1024):.1f} MB）超過系統限制（最大 50 MB）。",
        )

    doc_id = str(uuid.uuid4())

    try:
        chunks = DocumentParser.parse_document(
            file_bytes=file_bytes,
            filename=clean_filename,
            doc_id=doc_id,
            chunk_size=600,
            chunk_overlap=100,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件解析失敗：{str(e)}")

    if not chunks:
        raise HTTPException(
            status_code=400, detail="未能從文件中提取任何有效文字段落。"
        )

    chunk_texts = [c["text"] for c in chunks]
    try:
        embeddings = await AIClient.create_embeddings(
            base_url=base_url,
            api_key=api_key,
            embedding_model=embedding_model,
            texts=chunk_texts,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"向量化（Embedding）產生失敗，請確認 Embedding 模型名稱與 Base URL 設定是否正確：{str(e)}",
        )

    try:
        vector_store.add_document_chunks(
            doc_id=doc_id,
            filename=clean_filename,
            file_size=file_size,
            embedding_model=embedding_model,
            chunks=chunks,
            embeddings=embeddings,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"向量資料庫儲存失敗：{str(e)}"
        )

    return {
        "success": True,
        "message": f"成功解析並索引文件《{clean_filename}》",
        "doc_id": doc_id,
        "filename": clean_filename,
        "chunks_count": len(chunks),
    }


@app.get("/api/documents")
async def list_documents():
    return {"documents": vector_store.list_documents()}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    success = vector_store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="找不到指定的文件或刪除失敗。")
    return {"success": True, "message": "文件與向量索引已成功移除。"}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    base_url: str
    api_key: Optional[str] = None
    chat_model: str
    embedding_model: str
    top_k: int = 4
    system_prompt: Optional[str] = None
    history: Optional[List[ChatMessage]] = Field(default_factory=list)


DEFAULT_SYSTEM_INSTRUCTION = (
    "你是一位具備頂尖洞察力、嚴密邏輯與跨領域分析能力的「資深文檔專家與智庫顧問」。\n"
    "你的職責是深入研讀、穿透並分析提供的參考文件內容，為使用者的提問提供深刻、結構化且具實質決策價值的專業解答。\n\n"
    "【核心分析與思考框架】\n"
    "1. 核心意圖穿透（Intent & Synthesis）：\n"
    "   - 不要停留在機械式的字面比對；深入理解使用者問題背後的本質目的（例如：評估業務風險、技術架構可行性、關鍵數據對比、合約條款陷阱、或總結核心論點）。\n"
    "   - 進行跨段落與跨來源綜合推理（Cross-Context Fusion）：靈活串聯分散在不同章節或頁面的零散資訊，理清其前後脈絡、因果關聯或潛在矛盾。\n\n"
    "2. 嚴謹求實與智力推論的平衡：\n"
    "   - 嚴格奠基於事實：數據、專案名稱、法定條款、核心結論等，必須忠實於參考文檔，絕不可無中生有或捏造事實。\n"
    "   - 允許專業邏輯演繹：若使用者詢問背後意涵、優劣分析或後續建議，請在文檔確鑿事實的基礎上給出專業推論，並明確說明推論脈絡（例如「根據文件第 X 頁...，可進一步評估...」）。\n"
    "   - 誠實指明文檔盲點：若文檔確實未提及某細節，請精確說明文檔缺乏哪方面的資料，並依現有文檔最相關的篇章給予建設性的替代指引。\n\n"
    "【專業呈現與排版準則】\n"
    "1. 【核心結論】（TL;DR）：在回答的最開頭，用 1~2 句話直接提煉最核心的結論或答案要點，讓讀者第一眼即抓住關鍵。\n"
    "2. 【結構化深度剖析】：層次分明地拆解細節，使用清晰的數字序號、項目符號與重點加粗。\n"
    "3. 【主動善用對比表格】：當涉及方案評估、多個實體比較、優缺點分析、數據變更時，優先使用 Markdown 表格直觀呈現。\n"
    "4. 【專家洞察與建議】：在適當情境下，於文末補充專業洞見（如潛在風險、實施注意事項、或值得追蹤的關鍵指標）。\n\n"
    "【引用出處標註規範（Citations）】\n"
    "- 每當引述或參考了特定文件段落的內容、數據或論點時，必須自然地在該語句後方附上對應的來源標籤，例如 [來源 1] 或 [來源 2]，保持嚴謹透明。\n\n"
    "【語言與風格】\n"
    "- 請全篇使用流暢、自然且符合台灣習慣用語的標準繁體中文（如：資訊、專案、最佳化、架構、程式）。\n"
    "- 保持自信專業、條理清晰且具洞察力的智庫顧問語調。"
)


def _build_rag_prompt_messages(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    history: Optional[List[ChatMessage]] = None,
    custom_system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks):
        ref_tag = f"[來源 {idx + 1}] 《{chunk['filename']}》 (第 {chunk['page_number']} 頁)"
        context_blocks.append(f"{ref_tag}:\n{chunk['text']}")

    context_str = (
        "\n\n---\n\n".join(context_blocks)
        if context_blocks
        else "（未檢索到相關文件段落）"
    )

    system_instruction = (
        custom_system_prompt.strip()
        if custom_system_prompt and custom_system_prompt.strip()
        else DEFAULT_SYSTEM_INSTRUCTION
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_instruction}
    ]

    # Include recent chat history (up to last 6 turns)
    if history:
        for msg in history[-6:]:
            messages.append({"role": msg.role, "content": msg.content})

    user_prompt = (
        "請仔細研讀以下已檢索的文件參考段落，並結合整體對話脈絡，以顧問級的深度為使用者提供清晰、精準且具洞察力的專業解答：\n\n"
        f"=================== 參考文件內容 ===================\n"
        f"{context_str}\n"
        f"===================================================\n\n"
        f"【使用者的問題】：\n{question}\n\n"
        "請依據上述參考內容，給出條理清晰的分析解答（引用事實或數據處請自然標記 [來源 X]）："
    )
    messages.append({"role": "user", "content": user_prompt})
    return messages


@app.post("/api/chat")
async def chat_with_docs(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="請輸入問題內容。")

    try:
        query_embeddings = await AIClient.create_embeddings(
            base_url=req.base_url,
            api_key=req.api_key,
            embedding_model=req.embedding_model,
            texts=[question],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"問題向量化失敗，請檢查 Embedding 模型設定：{str(e)}",
        )

    if not query_embeddings:
        raise HTTPException(status_code=500, detail="向量化回傳為空。")

    retrieved_chunks = vector_store.query_similar_chunks(
        query_embedding=query_embeddings[0],
        embedding_model=req.embedding_model,
        query_text=question,
        doc_id=req.doc_id,
        top_k=req.top_k,
    )

    if not retrieved_chunks:
        return {
            "answer": "目前知識庫中尚未找到相關的文件內容。請確認是否已上傳文件，或嘗試調整問題關鍵字。",
            "citations": [],
        }

    messages = _build_rag_prompt_messages(
        question, retrieved_chunks, req.history, req.system_prompt
    )

    try:
        answer = await AIClient.chat_completion(
            base_url=req.base_url,
            api_key=req.api_key,
            chat_model=req.chat_model,
            messages=messages,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"語言模型生成失敗，請檢查 Chat 模型名稱與 API 金鑰：{str(e)}",
        )

    return {
        "answer": answer,
        "citations": retrieved_chunks,
    }


@app.post("/api/chat/stream")
async def chat_with_docs_stream(req: ChatRequest):
    """Server-Sent Events (SSE) streaming endpoint for instantaneous typewriter output."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="請輸入問題內容。")

    try:
        query_embeddings = await AIClient.create_embeddings(
            base_url=req.base_url,
            api_key=req.api_key,
            embedding_model=req.embedding_model,
            texts=[question],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"問題向量化失敗，請檢查 Embedding 模型設定：{str(e)}",
        )

    retrieved_chunks = vector_store.query_similar_chunks(
        query_embedding=query_embeddings[0],
        embedding_model=req.embedding_model,
        query_text=question,
        doc_id=req.doc_id,
        top_k=req.top_k,
    )

    messages = _build_rag_prompt_messages(
        question, retrieved_chunks, req.history, req.system_prompt
    )

    async def event_generator():
        # First send citation metadata
        citations_json = json.dumps(retrieved_chunks, ensure_ascii=False)
        yield f"event: citations\ndata: {citations_json}\n\n"

        if not retrieved_chunks:
            no_info_json = json.dumps(
                {
                    "delta": (
                        "目前知識庫中尚未找到相關的文件內容。請確認是否已上傳文件，或嘗試調整問題關鍵字。"
                    )
                },
                ensure_ascii=False,
            )
            yield f"event: message\ndata: {no_info_json}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        try:
            async for token in AIClient.chat_completion_stream(
                base_url=req.base_url,
                api_key=req.api_key,
                chat_model=req.chat_model,
                messages=messages,
            ):
                token_json = json.dumps({"delta": token}, ensure_ascii=False)
                yield f"event: message\ndata: {token_json}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
        except Exception as e:
            err_json = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {err_json}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
