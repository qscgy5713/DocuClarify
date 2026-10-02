import io
import re
from typing import Any, Dict, List
from pypdf import PdfReader


class DocumentParser:
    """Parses PDF, Markdown, and plain text files into structured chunks with

    page numbers and source tracking metadata.
    """

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> List[str]:
        """Splits a body of text into overlapping character chunks while

        attempting to break naturally on sentence or paragraph boundaries.
        """
        cleaned_text = re.sub(r"\r\n", "\n", text).strip()
        if not cleaned_text:
            return []

        chunks: List[str] = []
        start = 0
        total_len = len(cleaned_text)

        while start < total_len:
            end = start + chunk_size
            if end >= total_len:
                chunk = cleaned_text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Try to break on paragraph or sentence ending within the window
            boundary = cleaned_text.rfind("\n\n", start, end)
            if boundary == -1 or boundary <= start + chunk_size // 2:
                boundary = cleaned_text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start + chunk_size // 2:
                # Look for sentence endings (e.g., . ! ? 。 ；)
                match = re.search(r"[.。!?！？\n]", cleaned_text[start:end])
                if match:
                    sentence_ends = [
                        m.end()
                        for m in re.finditer(
                            r"[.。!?！？\n]", cleaned_text[start:end]
                        )
                    ]
                    if (
                        sentence_ends
                        and sentence_ends[-1] > chunk_size // 2
                    ):
                        boundary = start + sentence_ends[-1]

            if boundary != -1 and boundary > start:
                chunk = cleaned_text[start:boundary].strip()
                start = max(boundary, start + chunk_size - chunk_overlap)
            else:
                chunk = cleaned_text[start:end].strip()
                start = end - chunk_overlap

            if chunk:
                chunks.append(chunk)

        return chunks

    @staticmethod
    def stitch_text(text_a: str, text_b: str, max_overlap: int = 150) -> str:
        """Stitches two adjacent chunks text_a and text_b by identifying and

        deduplicating overlapping boundary characters.
        """
        clean_a = text_a.rstrip()
        clean_b = text_b.lstrip()

        overlap_window = min(len(clean_a), len(clean_b), max_overlap)
        # Check from longest possible overlap down to 10 characters
        for k in range(overlap_window, 10, -1):
            suffix = clean_a[-k:]
            if clean_b.startswith(suffix):
                return clean_a + clean_b[k:]

        # If no direct substring overlap found, join naturally with a newline
        return f"{clean_a}\n{clean_b}"

    @classmethod
    def merge_adjacent_chunks(
        cls,
        chunks: List[Dict[str, Any]],
        max_merged: int = 4,
    ) -> List[Dict[str, Any]]:
        """Detects chunks belonging to the same document with consecutive

        chunk_indices and seamlessly stitches them into unified, non-redundant
        context blocks.
        """
        if not chunks:
            return []

        # Group by doc_id
        by_doc: Dict[str, List[Dict[str, Any]]] = {}
        for c in chunks:
            by_doc.setdefault(c["doc_id"], []).append(c)

        merged_results: List[Dict[str, Any]] = []

        for doc_id, doc_chunks in by_doc.items():
            # Sort by chunk_index
            doc_chunks.sort(key=lambda x: x["chunk_index"])

            current = None
            for chunk in doc_chunks:
                if current is None:
                    current = {
                        "text": chunk["text"],
                        "doc_id": chunk["doc_id"],
                        "filename": chunk["filename"],
                        "page_start": chunk["page_number"],
                        "page_end": chunk["page_number"],
                        "chunk_indices": [chunk["chunk_index"]],
                        "score": chunk["score"],
                    }
                else:
                    last_idx = current["chunk_indices"][-1]
                    # If consecutive, stitch them together
                    if chunk["chunk_index"] == last_idx + 1:
                        current["text"] = cls.stitch_text(
                            current["text"], chunk["text"]
                        )
                        current["page_end"] = max(
                            current["page_end"], chunk["page_number"]
                        )
                        current["chunk_indices"].append(chunk["chunk_index"])
                        current["score"] = max(current["score"], chunk["score"])
                    else:
                        merged_results.append(current)
                        current = {
                            "text": chunk["text"],
                            "doc_id": chunk["doc_id"],
                            "filename": chunk["filename"],
                            "page_start": chunk["page_number"],
                            "page_end": chunk["page_number"],
                            "chunk_indices": [chunk["chunk_index"]],
                            "score": chunk["score"],
                        }
            if current:
                merged_results.append(current)

        final_chunks: List[Dict[str, Any]] = []
        for m in merged_results:
            p_start = m["page_start"]
            p_end = m["page_end"]
            page_display = (
                str(p_start) if p_start == p_end else f"{p_start}–{p_end}"
            )
            indices_display = "-".join(str(idx) for idx in m["chunk_indices"])

            final_chunks.append(
                {
                    "text": m["text"],
                    "doc_id": m["doc_id"],
                    "filename": m["filename"],
                    "page_number": page_display,
                    "chunk_index": indices_display,
                    "score": m["score"],
                    "is_merged": len(m["chunk_indices"]) > 1,
                }
            )

        # Sort by highest score first and limit to max_merged
        final_chunks.sort(key=lambda x: x["score"], reverse=True)
        return final_chunks[:max_merged]

    @classmethod
    def parse_pdf(
        cls,
        file_bytes: bytes,
        filename: str,
        doc_id: str,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        """Extracts text page-by-page from a PDF and produces chunk records with

        page numbers.
        """
        reader = PdfReader(io.BytesIO(file_bytes))
        chunks_with_metadata: List[Dict[str, Any]] = []
        global_chunk_idx = 0

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            page_text = page.extract_text() or ""
            page_chunks = cls.chunk_text(
                page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

            for chunk in page_chunks:
                chunks_with_metadata.append(
                    {
                        "id": f"{doc_id}_{global_chunk_idx}",
                        "doc_id": doc_id,
                        "filename": filename,
                        "page_number": page_num,
                        "chunk_index": global_chunk_idx,
                        "text": chunk,
                    }
                )
                global_chunk_idx += 1

        return chunks_with_metadata

    @classmethod
    def parse_text_or_markdown(
        cls,
        file_bytes: bytes,
        filename: str,
        doc_id: str,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        """Parses TXT or Markdown file content into chunk records."""
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="ignore")

        raw_chunks = cls.chunk_text(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks_with_metadata: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(raw_chunks):
            page_num = (idx // 3) + 1
            chunks_with_metadata.append(
                {
                    "id": f"{doc_id}_{idx}",
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_number": page_num,
                    "chunk_index": idx,
                    "text": chunk,
                }
            )

        return chunks_with_metadata

    @classmethod
    def parse_document(
        cls,
        file_bytes: bytes,
        filename: str,
        doc_id: str,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ) -> List[Dict[str, Any]]:
        lower_name = filename.lower()
        if lower_name.endswith(".pdf"):
            return cls.parse_pdf(
                file_bytes, filename, doc_id, chunk_size, chunk_overlap
            )
        else:
            return cls.parse_text_or_markdown(
                file_bytes, filename, doc_id, chunk_size, chunk_overlap
            )
