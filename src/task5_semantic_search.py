"""
Task 5 — Semantic Search Module.
"""

from src.index_store import (
    get_chroma_collection,
    get_embedding_model,
    index_exists,
    reset_index_cache,
)
from src.task4_chunking_indexing import run_pipeline


def _get_searchable_collection():
    """Return a Chroma collection, rebuilding the index if handles are stale."""
    if not index_exists():
        run_pipeline()

    try:
        collection = get_chroma_collection()
        collection.count()
        return collection
    except Exception:
        reset_index_cache()
        run_pipeline()
        return get_chroma_collection(force_refresh=True)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Dense retrieval using ChromaDB cosine similarity.
    """
    try:
        collection = _get_searchable_collection()
    except Exception:
        return []

    if collection.count() == 0:
        return []

    model = get_embedding_model()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict] = []
    if not results["documents"] or not results["documents"][0]:
        return output

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Chroma returns L2 distance; with normalized vectors: cosine_sim ≈ 1 - dist²/2
        score = float(max(0.0, 1.0 - (distance ** 2) / 2.0))
        output.append(
            {
                "content": doc,
                "score": score,
                "metadata": meta or {},
            }
        )

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
