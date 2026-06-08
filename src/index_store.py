"""
Shared index persistence for Tasks 4–6.

Uses ChromaDB (local persistent) for dense retrieval and a pickled BM25 corpus
for lexical search. Both are built by task4_chunking_indexing.run_pipeline().
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
INDEX_DIR = PROJECT_DIR / "data" / "index"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25_corpus.pkl"
META_PATH = INDEX_DIR / "meta.json"

COLLECTION_NAME = "DrugLawDocs"

_chroma_collection = None
_bm25_index = None
_bm25_corpus: list[dict] = []


def ensure_index_dir() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def save_meta(**kwargs) -> None:
    ensure_index_dir()
    META_PATH.write_text(json.dumps(kwargs, indent=2), encoding="utf-8")


def load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


def save_bm25_corpus(corpus: list[dict], bm25_index) -> None:
    ensure_index_dir()
    global _bm25_index, _bm25_corpus
    _bm25_corpus = corpus
    _bm25_index = bm25_index
    with BM25_PATH.open("wb") as f:
        pickle.dump({"corpus": corpus, "bm25": bm25_index}, f)


def load_bm25():
    global _bm25_index, _bm25_corpus
    if _bm25_index is not None:
        return _bm25_index, _bm25_corpus
    if not BM25_PATH.exists():
        return None, []
    with BM25_PATH.open("rb") as f:
        data = pickle.load(f)
    _bm25_corpus = data["corpus"]
    _bm25_index = data["bm25"]
    return _bm25_index, _bm25_corpus


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    import chromadb
    from chromadb.config import Settings

    ensure_index_dir()
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    _chroma_collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _chroma_collection


def get_embedding_model():
    meta = load_meta()
    model_name = meta.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def index_exists() -> bool:
    return BM25_PATH.exists() and any(CHROMA_DIR.iterdir()) if CHROMA_DIR.exists() else False
