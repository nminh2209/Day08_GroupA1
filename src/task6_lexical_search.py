"""
Task 6 — Lexical Search Module (BM25).
"""

import re
import unicodedata

import numpy as np

from src.index_store import index_exists, load_bm25
from src.task4_chunking_indexing import run_pipeline


def _normalize_vietnamese(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_vietnamese(text)
    return re.findall(r"\w+", normalized)


def _ensure_bm25():
    bm25, corpus = load_bm25()
    if bm25 is None or not corpus:
        if not index_exists():
            run_pipeline()
        bm25, corpus = load_bm25()
    return bm25, corpus


def build_bm25_index(corpus: list[dict]):
    """Build BM25 index from corpus."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Keyword search using BM25."""
    bm25, corpus = _ensure_bm25()
    if bm25 is None or not corpus:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict] = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue
        results.append(
            {
                "content": corpus[idx]["content"],
                "score": score,
                "metadata": corpus[idx].get("metadata", {}),
            }
        )

    return results[:top_k]


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
