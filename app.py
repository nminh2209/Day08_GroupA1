import streamlit as st
import os
import json
import re
from pathlib import Path
from src.task10_generation import generate_with_citation, get_openai_client, is_openai_configured
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import reorder_for_llm, format_context, build_sources_list, call_llm, extract_citations, _generate_fallback_answer, OPENAI_MODEL

st.set_page_config(page_title="RAG Chatbot - Pháp Luật Ma Tuý", page_icon="⚖️", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
    <style>
    /* Use semi-transparent background and inherited text color to support both Light and Dark mode */
    .metric-badge {
        background-color: rgba(128, 128, 128, 0.2);
        color: inherit;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-right: 5px;
        border: 1px solid rgba(128, 128, 128, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚖️ Trợ Lý Pháp Luật & Tin Tức Ma Tuý")
st.markdown("Hệ thống hỏi đáp ứng dụng RAG (Retrieval-Augmented Generation) với nguồn dữ liệu từ Pháp luật Việt Nam và Báo chí.")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Cài Đặt")
    use_hyde = st.checkbox("Bật HyDE (Hypothetical Document Embeddings)", value=False, help="Sinh ra tài liệu giả định để tăng cường độ chính xác khi tìm kiếm.")
    top_k = st.slider("Số lượng tài liệu truy xuất (Top K)", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    st.subheader("💡 Thông tin về Hệ thống")
    st.markdown("""
    - **Hybrid Search**: Kết hợp Semantic (Dense) và Lexical (BM25)
    - **Reranking**: Cross-encoder (Jina/Qwen)
    - **Fallback**: Vectorless search qua PageIndex
    - **HyDE**: Tùy chọn tăng cường ngữ nghĩa
    """)
    
    st.markdown("---")
    st.subheader("🛡️ Anti-Hallucination Guardrails")
    st.markdown("""
    Hệ thống tích hợp 3 lớp bảo vệ chống bịa đặt thông tin (Hallucination):
    1. **Context Threshold Filter**: Cắt bỏ nghiêm ngặt các văn bản có điểm số (score) < 0.3 ở bước Retrieval để loại bỏ thông tin rác.
    2. **Strict System Prompt**: Ép LLM chỉ được phép trả lời dựa trên context, phải trích dẫn nguồn cụ thể và cấm tuyệt đối tự sinh URL ngoại lai.
    3. **Citation Verification**: Post-processing đối chiếu các trích dẫn `[...]` của LLM với danh sách nguồn gốc. Sẽ bật cảnh báo ngay lập tức nếu phát hiện trích dẫn không có thật.
    """)

# Session State for Conversation Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📄 Xem Nguồn Tài Liệu"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['label']}** (Score: {src['score']:.3f})")
                    st.caption(f"{src['snippet']}...")

def generate_hypothetical_document(query: str) -> str:
    if not is_openai_configured():
        return query
    client = get_openai_client()
    system_prompt = "Bạn là một chuyên gia luật học. Hãy viết một đoạn văn ngắn (khoảng 3-4 câu) giải đáp trực tiếp giả định cho câu hỏi của người dùng. Đoạn văn này sẽ được dùng để tìm kiếm tài liệu pháp luật."
    try:
        res = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        return res.choices[0].message.content or query
    except Exception as e:
        return query

def custom_generate_with_citation(query: str, history: list, use_hyde: bool, top_k: int):
    # Contextualize query if history exists
    actual_query = query
    if history:
        # Simple context prepend for multi-turn
        actual_query = f"Dựa trên ngữ cảnh: {history[-1]['content']}, hãy trả lời: {query}"
    
    # Retrieval
    search_query = actual_query
    if use_hyde:
        hypo_doc = generate_hypothetical_document(actual_query)
        search_query = f"{actual_query}\n{hypo_doc}"
    
    chunks = retrieve(search_query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    sources = build_sources_list(chunks)

    model_used = None
    if is_openai_configured():
        answer = call_llm(actual_query, context)
        model_used = OPENAI_MODEL
    else:
        answer = _generate_fallback_answer(actual_query, reordered)

    citations = extract_citations(answer)
    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
        "model": model_used,
    }

# User Input
if prompt := st.chat_input("Nhập câu hỏi của bạn (VD: Hình phạt tàng trữ ma túy?)"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm tài liệu và tổng hợp câu trả lời..."):
            # Lấy history 2 lượt gần nhất (1 câu hỏi trước đó)
            history = st.session_state.messages[-3:-1] if len(st.session_state.messages) >= 3 else []
            
            result = custom_generate_with_citation(prompt, history, use_hyde, top_k)
            
            # 1. Highlight các trích dẫn trong câu trả lời
            formatted_answer = result["answer"]
            formatted_answer = re.sub(r'(\[[^\]]+\])', r'<span style="background-color: #ffeb3b; color: #000; padding: 2px 4px; border-radius: 4px; font-weight: bold;">\1</span>', formatted_answer)
            
            st.markdown(formatted_answer, unsafe_allow_html=True)
            
            # 2. Guardrail: Citation Verification
            valid_labels = [src["label"] for src in result["sources"]] + [f"Doc-{i+1}" for i in range(len(result["sources"]))]
            hallucinated_citations = []
            
            for cite in result["citations"]:
                # Trích dẫn LLM sinh ra có chứa ít nhất 1 nhãn tài liệu gốc thì hợp lệ
                is_valid = any(label in cite for label in valid_labels)
                if not is_valid:
                    hallucinated_citations.append(f"[{cite}]")
                    
            if hallucinated_citations:
                st.warning(f"⚠️ **Guardrail Cảnh Báo (Anti-Hallucination):** Phát hiện trích dẫn có thể bịa đặt: **{', '.join(hallucinated_citations)}**. Các nguồn này KHÔNG tồn tại trong tài liệu trích xuất. Vui lòng kiểm tra lại!")
            else:
                st.success("✅ **Guardrail Check:** Toàn bộ trích dẫn đã được tự động xác minh trùng khớp với danh sách tài liệu gốc.")

            
            # Display source metrics
            st.markdown(f"""
            <span class="metric-badge">Tài liệu: {len(result['sources'])}</span>
            <span class="metric-badge">Nguồn: {result['retrieval_source']}</span>
            <span class="metric-badge">Model: {result['model']}</span>
            """, unsafe_allow_html=True)

            if result["sources"]:
                with st.expander("📄 Nguồn Tài Liệu Tham Khảo (Click để xem chi tiết)"):
                    for src in result["sources"]:
                        st.markdown(f"**{src['label']}** - Score: {src['score']:.3f} ({src['retrieval']})")
                        st.info(f"{src['snippet']}...")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["answer"],
                "sources": result["sources"]
            })
