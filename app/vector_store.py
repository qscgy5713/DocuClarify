import json
import os
import re
import time
from typing import Any, Dict, List, Optional
import chromadb


class VectorStore:
    """Persistent in-process ChromaDB vector store for DocuClarify.

    Features collection isolation per embedding model, BM25 + Vector Hybrid
    Search, and automatic adjacent chunk stitching.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = (
            data_dir
            or os.getenv("DATA_DIR")
            or os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data")
            )
        )
        self.chroma_path = os.path.join(self.data_dir, "chroma")
        self.docs_metadata_path = os.path.join(
            self.data_dir, "documents_registry.json"
        )
        os.makedirs(self.chroma_path, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.chroma_path)

    def _get_collection_name(self, embedding_model: str) -> str:
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", embedding_model.strip())
        return f"docu_{safe_name}"

    def get_collection(self, embedding_model: str):
        collection_name = self._get_collection_name(embedding_model)
        return self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def _read_registry(self) -> Dict[str, Any]:
        if os.path.exists(self.docs_metadata_path):
            try:
                with open(self.docs_metadata_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _write_registry(self, data: Dict[str, Any]) -> None:
        with open(self.docs_metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_document_chunks(
        self,
        doc_id: str,
        filename: str,
        file_size: int,
        embedding_model: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        """Stores chunk texts, embeddings, and metadata into a model-isolated

        ChromaDB collection and registers document metadata.
        """
        if not chunks or not embeddings:
            return

        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "doc_id": chunk["doc_id"],
                "filename": chunk["filename"],
                "page_number": int(chunk["page_number"]),
                "chunk_index": int(chunk["chunk_index"]),
                "embedding_model": embedding_model,
            }
            for chunk in chunks
        ]

        collection = self.get_collection(embedding_model)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        max_page = max(c["page_number"] for c in chunks) if chunks else 1
        registry = self._read_registry()
        registry[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "file_size": file_size,
            "embedding_model": embedding_model,
            "chunks_count": len(chunks),
            "page_count": max_page,
            "created_at": int(time.time()),
        }
        self._write_registry(registry)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns the list of all registered documents."""
        registry = self._read_registry()
        docs = list(registry.values())
        docs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        return docs

    def delete_document(self, doc_id: str) -> bool:
        """Deletes all chunks of a document from its associated ChromaDB collection

        and registry.
        """
        registry = self._read_registry()
        doc_info = registry.get(doc_id)
        if doc_info:
            embedding_model = doc_info.get("embedding_model")
            if embedding_model:
                try:
                    collection = self.get_collection(embedding_model)
                    collection.delete(where={"doc_id": doc_id})
                except Exception:
                    pass
            del registry[doc_id]
            self._write_registry(registry)
            return True

        try:
            for coll in self.client.list_collections():
                try:
                    coll.delete(where={"doc_id": doc_id})
                except Exception:
                    pass
            return True
        except Exception:
            return False

    def get_all_chunks(
        self, embedding_model: str, doc_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetches all chunk texts and metadata from the collection to build BM25 sparse index."""
        try:
            collection = self.get_collection(embedding_model)
        except Exception:
            return []

        where_filter = {"doc_id": doc_id} if doc_id else None
        try:
            data = collection.get(
                where=where_filter, include=["documents", "metadatas"]
            )
        except Exception:
            return []

        if not data or not data.get("documents"):
            return []

        chunks: List[Dict[str, Any]] = []
        docs = data["documents"]
        metas = data.get("metadatas") or []
        ids = data.get("ids") or []

        for i, text in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"
            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "doc_id": meta.get("doc_id", ""),
                    "filename": meta.get("filename", "未知文件"),
                    "page_number": int(meta.get("page_number", 1)),
                    "chunk_index": int(meta.get("chunk_index", 0)),
                }
            )
        return chunks

    def query_similar_chunks(
        self,
        query_embedding: List[float],
        embedding_model: str,
        query_text: Optional[str] = None,
        doc_id: Optional[str] = None,
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """Performs Hybrid Search (Dense Vector + Sparse BM25 via Reciprocal Rank

        Fusion)
        and stitches adjacent continuous chunks.
        """
        try:
            collection = self.get_collection(embedding_model)
        except Exception:
            return []

        where_filter = {"doc_id": doc_id} if doc_id else None
        candidate_k = max(top_k * 2, 6)

        # 1. Dense vector search
        dense_retrieved: List[Dict[str, Any]] = []
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = (
                    results["metadatas"][0] if results.get("metadatas") else []
                )
                distances = (
                    results["distances"][0] if results.get("distances") else []
                )
                ids = results["ids"][0] if results.get("ids") else []

                for i, text in enumerate(docs):
                    meta = metas[i] if i < len(metas) else {}
                    dist = distances[i] if i < len(distances) else 1.0
                    score = round(1.0 - float(dist), 4)
                    dense_retrieved.append(
                        {
                            "id": ids[i] if i < len(ids) else None,
                            "text": text,
                            "doc_id": meta.get("doc_id", ""),
                            "filename": meta.get("filename", "未知文件"),
                            "page_number": int(meta.get("page_number", 1)),
                            "chunk_index": int(meta.get("chunk_index", 0)),
                            "score": score,
                        }
                    )
        except Exception:
            pass

        # 2. Sparse BM25 search (if query_text is provided)
        from app.bm25 import BM25Retriever, reciprocal_rank_fusion

        candidates = dense_retrieved
        if query_text:
            all_chunks = self.get_all_chunks(embedding_model, doc_id)
            if all_chunks:
                bm25 = BM25Retriever()
                bm25.index_documents(all_chunks)
                sparse_results = bm25.search(query_text, top_k=candidate_k)
                # Combine using Reciprocal Rank Fusion
                candidates = reciprocal_rank_fusion(
                    dense_results=dense_retrieved,
                    sparse_results=sparse_results,
                    rrf_k=60,
                    top_k=candidate_k,
                )

        # 3. Seamlessly stitch consecutive adjacent chunks into cohesive blocks
        from app.document_parser import DocumentParser

        return DocumentParser.merge_adjacent_chunks(
            candidates, max_merged=top_k
        )
