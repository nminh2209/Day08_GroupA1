"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.
"""

import re
import unicodedata

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search

SCORE_THRESHOLD = 0.05
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"


def _normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _apply_query_boost(query: str, results: list[dict]) -> list[dict]:
    """Boost chunks that match cited Điều numbers or named entities in the query."""
    if not results:
        return results

    article_nums = re.findall(r"Điều\s*(\d+)", query, flags=re.I)
    query_norm = _normalize(query)
    entity_tokens = [
        tok
        for tok in re.findall(r"[A-ZÀ-Ỵ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỵ][a-zà-ỹ]+)+", query)
        if len(tok) > 4
    ]

    boosted: list[dict] = []
    for item in results:
        score = float(item.get("score", 0.0))
        content_norm = _normalize(item.get("content", ""))

        for num in article_nums:
            if f"dieu {num}" in content_norm:
                score += 0.45

        for entity in entity_tokens:
            if _normalize(entity) in content_norm:
                score += 0.35

        boosted.append({**item, "score": score})

    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Full retrieval pipeline: hybrid search → RRF merge → rerank → PageIndex fallback.
    """
    pool_k = max(top_k * 3, 10)
    dense_results = _apply_query_boost(query, semantic_search(query, top_k=pool_k))
    sparse_results = _apply_query_boost(query, lexical_search(query, top_k=pool_k))

    if not dense_results and not sparse_results:
        return pageindex_search(query, top_k=top_k)

    merged = rerank_rrf([dense_results, sparse_results], top_k=pool_k)
    merged = _apply_query_boost(query, merged)
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    # Lọc bỏ các kết quả có điểm thấp hơn ngưỡng (tránh rác/hallucination)
    filtered_results = [r for r in final_results if r["score"] >= score_threshold]

    if not filtered_results:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    return filtered_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma túy năm 2024",
        "Luật phòng chống ma túy 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
