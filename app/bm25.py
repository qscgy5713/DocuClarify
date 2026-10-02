import math
import re
from typing import Any, Dict, List, Set, Tuple


class BM25Retriever:
    """Lightweight pure-Python BM25Okapi retriever with hybrid English & CJK

    n-gram tokenization.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenizes text into words for Latin text and unigrams + bigrams for CJK

        characters.
        """
        text = text.lower()
        tokens: List[str] = []

        # Extract alphanumeric words and alphanumeric codes (e.g. RFC-5545, v1.0, 100MB)
        words = re.findall(r"[a-z0-9_\-\.]+", text)
        tokens.extend(words)

        # Extract Chinese / Japanese / Korean characters
        cjk_chars = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text)
        # Add unigrams
        tokens.extend(cjk_chars)
        # Add bigrams for natural word boundary capture
        if len(cjk_chars) >= 2:
            bigrams = [
                cjk_chars[i] + cjk_chars[i + 1]
                for i in range(len(cjk_chars) - 1)
            ]
            tokens.extend(bigrams)

        return tokens

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Builds BM25 index from a list of chunk dictionaries with 'text' field."""
        self.corpus = documents
        self.doc_lengths = []
        self.doc_freqs = []
        df: Dict[str, int] = {}
        n_docs = len(documents)

        if n_docs == 0:
            self.avg_doc_len = 0.0
            self.idf = {}
            return

        for doc in documents:
            tokens = self.tokenize(doc.get("text", ""))
            self.doc_lengths.append(len(tokens))

            # Term frequency in current doc
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_freqs.append(tf)

            # Document frequency
            for t in tf.keys():
                df[t] = df.get(t, 0) + 1

        self.avg_doc_len = sum(self.doc_lengths) / n_docs

        # Calculate IDF with standard Robertson-Spärck Jones formula
        self.idf = {}
        for term, freq in df.items():
            # Add 0.5 smoothing
            self.idf[term] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(
        self, query: str, top_k: int = 6
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Scores all indexed documents against the query and returns top_k (doc,

        bm25_score).
        """
        if not self.corpus:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * len(self.corpus)

        for i, tf in enumerate(self.doc_freqs):
            doc_len = self.doc_lengths[i]
            len_norm = (
                1.0
                - self.b
                + self.b * (doc_len / (self.avg_doc_len if self.avg_doc_len else 1.0))
            )

            for token in query_tokens:
                if token in tf:
                    freq = tf[token]
                    idf = self.idf.get(token, 0.0)
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * len_norm
                    scores[i] += idf * (numerator / denominator)

        # Pair with documents and sort descending
        doc_scores = [
            (self.corpus[i], scores[i])
            for i in range(len(self.corpus))
            if scores[i] > 0.0
        ]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Tuple[Dict[str, Any], float]],
    rrf_k: int = 60,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    """Merges dense vector results and sparse BM25 results using Reciprocal Rank

    Fusion (RRF).
    """
    scores: Dict[str, float] = {}
    docs_by_id: Dict[str, Dict[str, Any]] = {}

    # Rank dense vector results
    for rank, doc in enumerate(dense_results):
        doc_key = doc.get("id") or f"{doc['doc_id']}_{doc['chunk_index']}"
        docs_by_id[doc_key] = doc
        scores[doc_key] = scores.get(doc_key, 0.0) + (1.0 / (rrf_k + rank + 1))

    # Rank sparse BM25 results
    for rank, (doc, _) in enumerate(sparse_results):
        doc_key = doc.get("id") or f"{doc['doc_id']}_{doc['chunk_index']}"
        if doc_key not in docs_by_id:
            docs_by_id[doc_key] = doc
        scores[doc_key] = scores.get(doc_key, 0.0) + (1.0 / (rrf_k + rank + 1))

    # Sort candidates by combined RRF score
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    fused_docs: List[Dict[str, Any]] = []
    for k in sorted_keys[:top_k]:
        doc_copy = dict(docs_by_id[k])
        # Include normalized score
        doc_copy["rrf_score"] = round(scores[k], 5)
        fused_docs.append(doc_copy)

    return fused_docs
