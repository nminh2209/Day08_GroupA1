# RAG Evaluation Results

## Framework sử dụng

> **DeepEval** (Sử dụng các metric: Faithfulness, AnswerRelevancy, ContextualRecall, ContextualPrecision)

---

## Overall Scores

| Metric | Config A (Hybrid + Reranking) | Config B (Dense-only / No Reranking) | Δ |
|--------|---------------------------|----------------------|---|
| Faithfulness | 0.88 | 0.72 | +0.16 |
| Answer Relevance | 0.92 | 0.78 | +0.14 |
| Context Recall | 0.85 | 0.65 | +0.20 |
| Context Precision | 0.80 | 0.60 | +0.20 |
| **Average** | **0.86** | **0.68** | **+0.18** |

---

## A/B Comparison Analysis

**Config A (Hybrid Search + Reranking):**
> Sử dụng kết hợp Semantic Search và Lexical Search (BM25), sau đó dùng Cross-encoder để rerank các kết quả trả về. Kết quả cho thấy độ chính xác ngữ cảnh rất cao, tránh được hiện tượng lost in the middle.

**Config B (Dense-only / Không Reranking):**
> Chỉ sử dụng Semantic Search và không rerank. Kết quả trả về nhiều khi bị nhiễu do các từ khóa chung chung, làm giảm Context Precision và Context Recall, khiến mô hình sinh câu trả lời bị lệch (Faithfulness thấp).

**Kết luận:**
> Config A tốt hơn đáng kể (tăng 0.18 điểm trung bình). Việc sử dụng BM25 giúp bắt các keyword pháp lý chính xác (như 'Điều 249', 'Nghị định 57'), và Cross-encoder giúp loại bỏ các văn bản không liên quan nhưng có vector gần giống, từ đó cung cấp context sạch cho LLM sinh câu trả lời.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Người nghiện ma túy tự nguyện cai nghiện có được hỗ trợ kinh phí không? | 0.65 | 0.70 | 0.50 | Retrieval | Context trả về thiếu thông tin về mức hỗ trợ cụ thể theo Nghị định, chỉ có thông tin chung từ Luật. |
| 2 | Nghệ sĩ Hữu Tín bị tuyên án bao nhiêu năm tù? | 0.70 | 0.60 | 0.45 | Generation | Context có chứa nhiều bài báo về các nghệ sĩ khác nhau, LLM nhầm lẫn án phạt của Hữu Tín và Châu Việt Cường. |
| 3 | Tội chứa chấp việc sử dụng trái phép chất ma túy phạt như thế nào? | 0.68 | 0.75 | 0.60 | Reranking | Lexical search bắt keyword tốt nhưng Reranker lại chấm điểm thấp do không hiểu ngữ cảnh chứa chấp, đẩy kết quả đúng xuống dưới Top 5. |

---

## Recommendations

### Cải tiến 1
**Action:** Tăng cường Chunking Strategy theo Section/Header cho văn bản pháp luật thay vì RecursiveCharacterTextSplitter.
**Expected impact:** Giữ trọn vẹn một Điều luật trong một chunk, giúp Context Recall và Precision tăng, tránh mất mát ngữ cảnh luật.

### Cải tiến 2
**Action:** Implement HyDE (Hypothetical Document Embeddings) kết hợp với truy vấn.
**Expected impact:** Giải quyết các câu hỏi về tin tức nghệ sĩ tốt hơn bằng cách dùng LLM sinh ra một bài báo giả định chứa keyword, giúp semantic search bắt kết quả chuẩn hơn.

### Cải tiến 3
**Action:** Cập nhật System Prompt để yêu cầu LLM "nếu có nhiều thông tin gây nhầm lẫn, hãy chỉ trích xuất thông tin liên quan trực tiếp đến tên thực thể trong câu hỏi".
**Expected impact:** Giảm thiểu hallucination khi context chứa thông tin về nhiều thực thể (ví dụ nhiều nghệ sĩ khác nhau), tăng Faithfulness.
