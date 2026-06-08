"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.
"""

import json
import re
from pathlib import Path

from markitdown import MarkItDown
import docx

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _clean_text_block(text: str) -> str:
    """Remove page-template noise and keep the real legal body text."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"https?://[^\s)]+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    seen = set()
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if any(token in line for token in ("THƯ VIỆN PHÁP LUẬT", "Trang chủ", "Liên hệ", "Danh mục", "Sơ đồ WebSite", "Đăng nhập", "Đăng ký", "Tư vấn Pháp luật", "Pháp Luật Nhà Đất", "Chủ đề liên quan", "Từ khóa:", "Lưu trữ", "Ghi chú", "Bài liên quan:", "PHÁP LUẬT DOANH NGHIỆP")):
            continue
        if line.startswith("* [") or line.startswith("- [") or line.startswith("[") and "https://" in line:
            continue
        if len(line) < 6:
            continue
        norm = re.sub(r"\s+", " ", line).lower()
        if norm in seen:
            continue
        seen.add(norm)
        lines.append(line)

    cleaned = "\n".join(lines)
    footer_idx = cleaned.find("**CHỦ TỊCH QUỐC HỘI")
    if footer_idx != -1:
        cleaned = cleaned[:footer_idx]

    # Keep the legal body from the first real article heading onward.
    for marker in ("**Chương I**", "**Điều 1. ", "**Điều 1.**", "Chương I", "Điều 1"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[idx:]
            break

    return cleaned.strip()


def _read_docx_text(filepath: Path) -> str:
    """Extract text from DOCX while filtering website-template noise."""
    document = docx.Document(filepath)
    paragraphs = [p.text for p in document.paragraphs]
    text = "\n".join(paragraphs)
    return _clean_text_block(text)


def convert_legal_docs() -> int:
    """Convert PDF/DOCX files in data/landing/legal/ to markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("⚠ Không tìm thấy data/landing/legal/")
        return 0

    md = MarkItDown()
    count = 0

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue
        print(f"Converting: {filepath.name}")
        if filepath.suffix.lower() in {".docx", ".doc"}:
            text = _read_docx_text(filepath)
        else:
            result = md.convert(str(filepath))
            text = result.text_content
        text = _clean_text_block(text)
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(text, encoding="utf-8")
        print(f"  ✓ Saved: {output_path} ({len(text)} chars)")
        count += 1

    return count


def convert_news_articles() -> int:
    """Convert JSON crawled articles in data/landing/news/ to markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("⚠ Không tìm thấy data/landing/news/")
        return 0

    count = 0
    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue
        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"

        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
        content = header + data.get("content_markdown", "")
        output_path.write_text(content, encoding="utf-8")
        print(f"  ✓ Saved: {output_path}")
        count += 1

    return count


def clean_stale_outputs() -> None:
    """Remove standardized files that no longer have a landing source."""
    landing_legal = {p.stem for p in (LANDING_DIR / "legal").glob("*.*") if p.is_file()}
    landing_news = {p.stem for p in (LANDING_DIR / "news").glob("*.json")}

    legal_out = OUTPUT_DIR / "legal"
    news_out = OUTPUT_DIR / "news"
    if legal_out.exists():
        for md in legal_out.glob("*.md"):
            if md.stem not in landing_legal:
                md.unlink()
                print(f"  [CLEAN] {md.name}")
    if news_out.exists():
        for md in news_out.glob("*.md"):
            if md.stem not in landing_news:
                md.unlink()
                print(f"  [CLEAN] {md.name}")


def convert_all(clean_stale: bool = True) -> None:
    """Convert all landing files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    if clean_stale:
        print("\n--- Clean stale outputs ---")
        clean_stale_outputs()

    print("\n--- Legal Documents ---")
    legal_count = convert_legal_docs()

    print("\n--- News Articles ---")
    news_count = convert_news_articles()

    print(f"\n✓ Done! Converted {legal_count} legal + {news_count} news files")
    print(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
