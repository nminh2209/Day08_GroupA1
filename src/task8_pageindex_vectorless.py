"""
Task 8 — PageIndex Vectorless RAG.

Uses PageIndex API when PAGEINDEX_API_KEY is set; otherwise falls back to
local keyword search over standardized markdown (for offline testing).
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents() -> None:
    """Upload markdown documents to PageIndex."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Set PAGEINDEX_API_KEY in .env to upload documents")

    try:
        from pageindex import PageIndex
    except ImportError as exc:
        raise ImportError("pip install pageindex") from exc

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        pi.upload(
            content=content,
            metadata={"filename": md_file.name, "type": md_file.parent.name},
        )
        print(f"  ✓ Uploaded: {md_file.name}")


def _local_vectorless_search(query: str, top_k: int = 5) -> list[dict]:
    """Offline fallback: score markdown chunks by keyword overlap."""
    if not STANDARDIZED_DIR.exists():
        return []

    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return []

    scored: list[tuple[float, dict]] = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            if len(para) < 50:
                continue
            doc_tokens = set(re.findall(r"\w+", para.lower()))
            overlap = len(query_tokens & doc_tokens)
            if overlap == 0:
                continue
            score = overlap / len(query_tokens)
            scored.append(
                (
                    score,
                    {
                        "content": para,
                        "score": score,
                        "metadata": {
                            "source": md_file.name,
                            "type": md_file.parent.name,
                            "chunk_index": i,
                        },
                        "source": "pageindex",
                    },
                )
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex API or local fallback.
    """
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex

            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            results = pi.query(query=query, top_k=top_k)
            return [
                {
                    "content": r.text,
                    "score": float(r.score),
                    "metadata": getattr(r, "metadata", {}) or {},
                    "source": "pageindex",
                }
                for r in results
            ]
        except Exception:
            pass

    return _local_vectorless_search(query, top_k=top_k)


if __name__ == "__main__":
    if PAGEINDEX_API_KEY:
        print("Uploading documents...")
        upload_documents()

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma túy", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
