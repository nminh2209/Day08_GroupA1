"""
Task 4 — Chunking & Indexing vào Vector Store.

Chunking: RecursiveCharacterTextSplitter — an toàn với văn bản pháp luật dài,
giữ đoạn văn theo paragraph trước khi cắt theo câu/từ.

Embedding: paraphrase-multilingual-MiniLM-L12-v2 — nhẹ, hỗ trợ tiếng Việt tốt,
384 chiều, chạy local không cần API.

Vector store: ChromaDB persistent — đơn giản, không cần Docker như Weaviate.
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.index_store import (
    COLLECTION_NAME,
    get_chroma_collection,
    save_bm25_corpus,
    save_meta,
)

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# 500 chars ≈ 1 đoạn luật ngắn; overlap 50 giữ ngữ cảnh giữa các chunk liền kề
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Multilingual, 384-dim — cân bằng tốc độ và chất lượng cho tiếng Việt
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

VECTOR_STORE = "chromadb"


def load_documents() -> list[dict]:
    """Load all markdown files from data/standardized/."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: list[dict] = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if not chunk_text.strip():
                continue
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": i},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer | None = None) -> list[dict]:
    """Embed chunks with sentence-transformers."""
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def _tokenize(text: str) -> list[str]:
    import re
    import unicodedata

    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.findall(r"\w+", text)


def index_to_vectorstore(chunks: list[dict], reset: bool = True) -> None:
    """Persist chunks to ChromaDB and build BM25 index."""
    import chromadb
    from chromadb.config import Settings

    from src.index_store import CHROMA_DIR, ensure_index_dir

    ensure_index_dir()

    if reset and CHROMA_DIR.exists():
        import shutil

        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        ensure_index_dir()

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk_{i}"
        ids.append(chunk_id)
        documents.append(chunk["content"])
        metadatas.append(
            {
                "source": chunk["metadata"].get("source", ""),
                "type": chunk["metadata"].get("type", ""),
                "chunk_index": chunk["metadata"].get("chunk_index", 0),
            }
        )
        embeddings.append(chunk["embedding"])

    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings[start:end],
        )

    tokenized = [_tokenize(c["content"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    save_bm25_corpus(chunks, bm25)

    save_meta(
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=EMBEDDING_DIM,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        chunking_method=CHUNKING_METHOD,
        vector_store=VECTOR_STORE,
        num_chunks=len(chunks),
    )


def run_pipeline() -> None:
    """Run load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    if not docs:
        print("⚠ Không có documents. Chạy Task 1–3 trước.")
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    model = SentenceTransformer(EMBEDDING_MODEL)
    chunks = embed_chunks(chunks, model=model)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to ChromaDB + BM25")


if __name__ == "__main__":
    run_pipeline()
