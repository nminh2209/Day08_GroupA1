"""
Task 10 — Generation Có Citation.

Pipeline:
    retrieve → reorder_for_llm → format_context → OpenAI → extract citations

Cấu hình trong .env:
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini   (optional, default gpt-4o-mini)
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_DIR = Path(__file__).parent.parent
load_dotenv(_PROJECT_DIR / ".env")

from src.task9_retrieval_pipeline import retrieve

# top_k=5: đủ evidence, tránh lost-in-the-middle với context quá dài
TOP_K = 5
# top_p=0.9: nucleus sampling — đủ đa dạng nhưng không quá ngẫu nhiên
TOP_P = 0.9
# temperature=0.3: RAG cần factual, ít hallucination
TEMPERATURE = 0.3

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source using the document labels provided in the context
(e.g., [Luật Phòng chống ma tuý 2021, Điều 3] or [VnExpress, 2023] or [Doc-1]).

If the information is not explicitly stated in the provided context,
state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation in brackets
- Use the Source label from each document block for citations
- If context is insufficient, say so clearly
- NEVER generate markdown links like `[text](http...)` or include external URLs under any circumstances
- Do NOT include footer text, registration form text, or navigational text from the sources
- Structure your answer with clear paragraphs"""


def is_openai_configured() -> bool:
    """Check if a real OpenAI API key is set in .env."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return False
    placeholders = {"sk-xxx", "sk-your-openai-key-here", "sk-your-key-here"}
    return key not in placeholders and key.startswith("sk-")


def get_openai_client():
    """Return an OpenAI client or raise if not configured."""
    if not is_openai_configured():
        raise RuntimeError(
            "OPENAI_API_KEY chưa được cấu hình. "
            "Sao chép .env.example thành .env và điền key của bạn."
        )
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_source_label(metadata: dict, index: int) -> str:
    """
    Build a human-readable citation label from chunk metadata.

    Examples:
        "Luật Phòng chống ma tuý 2021" (from luat-phong-chong-ma-tuy-2021.md)
        "VnExpress" (from news article)
    """
    source = metadata.get("source", f"Doc-{index}")
    doc_type = metadata.get("type", "")

    # Strip extension and prettify filename
    name = Path(source).stem
    name = name.replace("-", " ").replace("_", " ")

    if "luat" in name.lower() or "luật" in name.lower():
        return "Luật Phòng chống ma tuý 2021" if "2021" in name else name.title()
    if "nghi dinh" in name.lower() or "nghi-dinh" in name.lower():
        parts = name.split()
        for i, p in enumerate(parts):
            if p.isdigit() and i + 1 < len(parts):
                return f"Nghị định {p}/{parts[i + 1]}"
        return name.title()
    if "bo luat" in name.lower() or "bo-luat" in name.lower():
        return "Bộ luật Hình sự 2015"
    if doc_type == "news" or "vnexpress" in source.lower() or "tuoitre" in source.lower():
        if "vnexpress" in source.lower() or "ngoisao" in source.lower():
            return "VnExpress"
        if "tuoitre" in source.lower():
            return "Tuổi Trẻ"
        return "Báo chí"
    return name.title() if name else f"Doc-{index}"


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Reorder to mitigate lost-in-the-middle: best at start and end, weaker in middle.
    Input [1,2,3,4,5] → Output [1,3,5,4,2]
    """
    if len(chunks) <= 2:
        return chunks

    reordered: list[dict] = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    start = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for i in range(start, 0, -2):
        reordered.append(chunks[i])

    return reordered


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks into a context string with citation labels for the LLM.

    Each block is labeled [Doc-N | Source: ... | Type: ...] so the LLM
    can cite using the Source name in brackets.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
        label = format_source_label(metadata, i)
        context_parts.append(
            f"[Doc-{i} | Source: {label} | File: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


def build_sources_list(chunks: list[dict]) -> list[dict]:
    """Build structured source list from retrieved chunks."""
    sources = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        sources.append(
            {
                "doc_id": f"Doc-{i}",
                "label": format_source_label(metadata, i),
                "file": metadata.get("source", ""),
                "type": metadata.get("type", "unknown"),
                "score": chunk.get("score", 0.0),
                "retrieval": chunk.get("source", "hybrid"),
                "snippet": chunk["content"][:300],
            }
        )
    return sources


def extract_citations(answer: str) -> list[str]:
    """
    Extract citation strings from LLM answer.

    Matches patterns like [Luật Phòng chống ma tuý 2021, Điều 3] or [VnExpress, 2023].
    """
    return re.findall(r"\[([^\]]+)\]", answer)


def call_llm(query: str, context: str, model: str | None = None) -> str:
    """Call OpenAI chat completion with system prompt and context."""
    client = get_openai_client()
    model = model or OPENAI_MODEL
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or ""


def _generate_fallback_answer(query: str, chunks: list[dict]) -> str:
    """Rule-based answer when OpenAI key is not configured."""
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    label = format_source_label(chunks[0].get("metadata", {}), 1)
    snippet = chunks[0]["content"][:400].strip()
    return (
        f"Dựa trên tài liệu [{label}], thông tin liên quan đến câu hỏi "
        f"'{query}' như sau: {snippet}... [{label}]"
    )


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation with citations.

    Returns:
        {
            'answer': str,              # LLM answer with inline [citations]
            'sources': list[dict],      # Retrieved chunks used as context
            'citations': list[str],     # Citation strings extracted from answer
            'retrieval_source': str,    # 'hybrid' or 'pageindex'
            'model': str | None,        # OpenAI model used, or None if fallback
        }
    """
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    sources = build_sources_list(chunks)

    model_used = None
    if is_openai_configured():
        answer = call_llm(query, context)
        model_used = OPENAI_MODEL
    else:
        answer = _generate_fallback_answer(query, reordered)

    citations = extract_citations(answer)

    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "model": model_used,
    }


if __name__ == "__main__":
    print(f"OpenAI configured: {is_openai_configured()}")
    if not is_openai_configured():
        print("-> Dien OPENAI_API_KEY vao file .env de dung LLM generation")
        print(f"   File: {_PROJECT_DIR / '.env'}")

    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma túy?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma túy 2021?",
    ]

    for q in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\nCitations found: {result['citations']}")
        print(
            f"[Sources: {len(result['sources'])} chunks | "
            f"via {result['retrieval_source']} | model: {result['model']}]"
        )
