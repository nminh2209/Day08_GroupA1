"""
Task 7 — Reranking Module.

Default: cross-encoder via Jina API if key available, else keyword-overlap fallback.
Also implements MMR and RRF for fusion in Task 9.
"""

import os
import re
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def _keyword_overlap_score(query: str, document: str) -> float:
    """Simple fallback reranker based on keyword overlap."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    doc_tokens = set(re.findall(r"\w+", document.lower()))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    return overlap / len(query_tokens)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Rerank using Jina API or keyword-overlap fallback."""
    if not candidates:
        return []

    if JINA_API_KEY:
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": [c["content"] for c in candidates],
                    "top_n": min(top_k, len(candidates)),
                },
                timeout=30,
            )
            response.raise_for_status()
            reranked = response.json()["results"]
            results = []
            for item in reranked:
                candidate = {**candidates[item["index"]], "score": float(item["relevance_score"])}
                results.append(candidate)
            return results
        except requests.RequestException:
            pass

    scored = []
    for candidate in candidates:
        rerank_score = _keyword_overlap_score(query, candidate["content"])
        scored.append({**candidate, "score": rerank_score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va = np.array(a)
    vb = np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Maximal Marginal Relevance selection."""
    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            emb = candidates[idx].get("embedding")
            if emb:
                relevance = _cosine_sim(query_embedding, emb)
            else:
                relevance = candidates[idx].get("score", 0.0)

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sel_emb = candidates[sel_idx].get("embedding")
                if emb and sel_emb:
                    sim = _cosine_sim(emb, sel_emb)
                    max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [{**candidates[i], "score": candidates[i].get("score", 0.0)} for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """Unified reranking interface."""
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        from src.index_store import get_embedding_model

        model = get_embedding_model()
        query_emb = model.encode(query, normalize_embeddings=True).tolist()
        return rerank_mmr(query_emb, candidates, top_k=top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
