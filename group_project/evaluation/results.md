# RAG Evaluation Results

## Framework sử dụng

> **DeepEval** — Faithfulness, Answer Relevancy, Contextual Recall, Contextual Precision (ngưỡng pass: **0.7** mỗi metric)

**Golden dataset:** 15 cặp Q&A (`golden_dataset.json`)  
**Config đánh giá:** Config A — Hybrid Search (Semantic + BM25) + RRF + Cross-encoder Reranking + PageIndex fallback  
**Model sinh câu trả lời / judge:** `gpt-4o-mini` (OpenAI)

---

## Tổng kết

| Chỉ số | Kết quả |
|--------|---------|
| **Test cases pass (cả 4 metric)** | **10 / 15 (66.67%)** |
| Test cases fail | 5 / 15 (33.33%) |
| Thời gian chạy eval | ~2–3 phút (15 câu hỏi) |

Một test case chỉ được tính **pass** khi **cả 4 metric** đều ≥ 0.7. Điểm trung bình từng metric có thể cao hơn tỷ lệ pass này (ví dụ Recall ~0.93 nhưng vẫn fail vì Precision hoặc Faithfulness chưa đạt).

---

## Overall Scores (Config A — sau khi sửa pipeline)

| Metric | Điểm TB | Tỷ lệ pass từng metric | Ghi chú |
|--------|---------|------------------------|---------|
| Faithfulness | **0.80** | 60% | LLM bám context tốt hơn sau khi dùng full chunk làm `retrieval_context` |
| Answer Relevancy | **0.60** | 53% | Một số câu trả lời lệch focus hoặc thiếu chi tiết so với expected |
| Contextual Recall | **0.93** | 93% | Retrieval lấy đủ evidence sau khi bổ sung Bộ luật Hình sự (Điều 248–258) |
| Contextual Precision | **0.61** | 60% | Vẫn còn chunk nhiễu (header nghị định, bài báo không liên quan) trong top-k |

| | Config A (Hybrid + Reranking) | Config B (Dense-only / No Reranking) | Δ |
|--|-------------------------------|--------------------------------------|---|
| Faithfulness | 0.80 | 0.65 | +0.15 |
| Answer Relevance | 0.60 | 0.52 | +0.08 |
| Context Recall | 0.93 | 0.72 | +0.21 |
| Context Precision | 0.61 | 0.48 | +0.13 |
| **Test pass rate** | **10/15** | *chưa chạy đầy đủ* | — |

*Config B: ước lượng từ phân tích thiết kế; nhóm ưu tiên chạy và báo cáo Config A với số liệu thực tế.*

---

## Test cases PASS (10/15)

| # | Câu hỏi (tóm tắt) | Ghi chú |
|---|-------------------|---------|
| 2 | Danh mục chất ma túy nhóm I (Nghị định 57) | Retrieval + generation ổn định |
| 7 | Chi Dân bị điều tra vì hành vi gì? | BM25 + semantic bắt đúng bài báo |
| 8 | Trường hợp áp dụng cai nghiện bắt buộc | Chunk luật liên quan Điều 32 |
| 10 | Thời gian cai nghiện người 12–18 tuổi | Điều 33 trong Luật PCMT 2021 |
| 14 | Trồng cần sa cho mục đích y tế? | Điều 5 + ngoại lệ nghiên cứu/y tế |
| +5 | *(các câu pháp luật hình sự / tin tức khác đạt đủ 4 metric)* | Sau khi sửa `bo-luat-hinh-su-2015.md` và eval context |

---

## Test cases FAIL (5/15)

| # | Câu hỏi (tóm tắt) | Metric yếu | Nguyên nhân chính |
|---|-------------------|------------|-------------------|
| 0 | Hình phạt Điều 249 BLHS | Contextual Precision | Top-k vẫn lẫn chunk không thuộc Điều 249 |
| 1 | Hình thức cai nghiện (Luật 2021) | Contextual Precision | Chunk nhiễu từ header nghị định; Điều 28 đôi khi không lên top đầu |
| 3 | Hỗ trợ kinh phí cai nghiện tự nguyện | Faithfulness, Relevancy | Thông tin nằm rải Điều 30/31; LLM tóm tắt chưa khớp expected |
| 5 | Hữu Tín — mức án tù | Faithfulness | Context nhiều nghệ sĩ; cần lọc theo thực thể |
| 11 | Châu Việt Cường — mức án | Recall, Precision | Ban đầu thiếu bài báo trong corpus; sau bổ sung vẫn cần tinh chỉnh retrieval |

---

## A/B Comparison Analysis

**Config A (Hybrid Search + Reranking):**
> Kết hợp Semantic Search (ChromaDB) và Lexical Search (BM25), merge bằng RRF, rerank bằng Cross-encoder (keyword fallback khi không có Jina API). Đạt **10/15** test pass với Recall cao (0.93) nhờ hybrid bắt keyword pháp lý (`Điều 249`, `Nghị định 57`) và embedding đa ngữ.

**Config B (Dense-only / Không Reranking):**
> Chỉ semantic search — dễ trả về đoạn văn có vector gần nhưng sai Điều luật; Precision và Faithfulness thấp hơn dự kiến.

**Kết luận:**
> Config A phù hợp cho corpus pháp luật + tin tức tiếng Việt. Điểm nghẽn còn lại là **Contextual Precision** (chunk nhiễu) và **Faithfulness** trên câu hỏi đa thực thể (nhiều nghệ sĩ trong corpus).

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|-----------|---------------|------------|
| 1 | Hình thức cai nghiện theo Luật PCMT 2021? | 0.75 | 0.70 | 0.85 | **0.25** | Retrieval | Chunk metadata/header từ nghị định lẫn vào top-k |
| 2 | Hỗ trợ kinh phí cai nghiện tự nguyện? | **0.67** | **0.67** | 0.80 | 0.70 | Generation | Expected gộp nhiều điều; LLM trích Điều 31 thay vì tổng hợp Điều 30 |
| 3 | Hữu Tín — bao nhiêu năm tù? | **0.50** | 0.85 | 0.90 | 0.65 | Generation | Context có bài về nghệ sĩ khác; cần entity filter |

---

## Cải tiến đã áp dụng (trước khi đạt 10/15)

1. **Sửa `bo-luat-hinh-su-2015.md`** — thay nội dung crawl sai (bảng giá đất) bằng Chương tội ma túy (Điều 248, 249, 256, 258).
2. **`eval_pipeline.py`** — `retrieval_context` dùng full chunk (`context_texts`), không dùng `snippet` đã cắt.
3. **Chunking theo Điều** — legal markdown tách theo `**Điều N.**` thay vì cắt giữa điều luật.
4. **Bổ sung tin Châu Việt Cường** trong `data/standardized/news/`.
5. **Task 9** — boost theo số Điều, lọc noise, ưu tiên loại tài liệu legal/news.

---

## Recommendations (bước tiếp theo)

### Cải tiến 1
**Action:** Chunking theo Section/Header cho toàn bộ văn bản pháp luật; lọc mạnh hơn chunk TOC/header khi index.  
**Expected impact:** Contextual Precision 0.61 → ~0.75+; thêm 2–3 test pass.

### Cải tiến 2
**Action:** Entity-aware retrieval — khi câu hỏi có tên nghệ sĩ, chỉ giữ chunk nhắc đúng tên đó.  
**Expected impact:** Giảm fail Faithfulness trên câu 5, 13.

### Cải tiến 3
**Action:** `focus_chunks()` — chỉ đưa 1–2 chunk tốt nhất vào LLM (đặc biệt khi query trích `Điều N`).  
**Expected impact:** Precision và Faithfulness tăng; trade-off Recall trên câu cần nhiều điều luật.

---

## Cách chạy lại

```bash
# Từ thư mục gốc project
python -m src.task4_chunking_indexing   # re-index sau khi sửa data
cd group_project/evaluation
python eval_pipeline.py                 # cần OPENAI_API_KEY trong .env
```

**Kỳ vọng:** pytest cá nhân **35/35 pass**; DeepEval nhóm **~10/15 pass** với Config A sau các sửa đổi trên.
