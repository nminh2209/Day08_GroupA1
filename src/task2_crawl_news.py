"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Dùng Crawl4AI để crawl bài báo thật từ VnExpress, Tuổi Trẻ, Ngôi Sao.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from src.crawl_utils import DEFAULT_CRAWL_CONFIG

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Bài báo thật — đã xác minh tồn tại
ARTICLE_URLS = [
    # VnExpress
    "https://vnexpress.net/dien-vien-le-hang-bi-dieu-tra-mua-ban-ma-tuy-4597048.html",
    "https://vnexpress.net/dien-vien-hai-huu-tin-su-dung-ma-tuy-vi-to-mo-4599355.html",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-va-ca-si-chi-dan-bi-bat-4814295.html",
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://vnexpress.net/ca-si-chu-bin-bi-tam-giu-vi-lien-quan-ma-tuy-4755275.html",
    "https://vnexpress.net/su-nghiep-long-nhat-truoc-khi-bi-bat-vi-lien-quan-ma-tuy-5076081.html",
    "https://ngoisao.vnexpress.net/chi-dan-va-andrea-aybar-bi-khoi-to-vi-to-chuc-su-dung-ma-tuy-4815983.html",
    # Tuổi Trẻ
    "https://tuoitre.vn/bat-ca-si-long-nhat-va-ca-si-son-ngoc-minh-vi-lien-quan-ma-tuy-20260520082138943.htm",
    "https://tuoitre.vn/ca-si-long-nhat-thua-nhan-da-nhieu-lan-dat-mua-ma-tuy-ve-su-dung-20260520161117184.htm",
    "https://tuoitre.vn/vu-miu-le-long-nhat-son-ngoc-minh-nghe-si-phai-giu-hinh-anh-chin-chu-tren-san-khau-lan-ngoai-doi-2026052112085492.htm",
]


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _url_slug(url: str) -> str:
    """Stable filename slug from article URL path."""
    path = urlparse(url).path.strip("/")
    slug = path.split("/")[-1]
    slug = re.sub(r"\.(html?|htm)$", "", slug, flags=re.I)
    slug = re.sub(r"[^\w-]", "-", slug.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:80] or "article"


async def crawl_article(url: str) -> dict:
    """Crawl one news article and return metadata + content."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(url=url, config=DEFAULT_CRAWL_CONFIG)

    if not result.success:
        raise RuntimeError(f"Crawl that bai: {url} - {result.error_message}")

    title = "Unknown"
    if result.metadata:
        title = (
            result.metadata.get("title")
            or result.metadata.get("og:title")
            or title
        )

    content = result.markdown or result.cleaned_html or ""
    if len(content.strip()) < 200:
        raise RuntimeError(
            f"Noi dung crawl qua ngan ({len(content)} chars) tu {url}."
        )

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


async def crawl_all(clean_first: bool = True) -> list[Path]:
    """Crawl all articles in ARTICLE_URLS."""
    setup_directory()

    if clean_first:
        for old in DATA_DIR.glob("*.json"):
            old.unlink()

    saved: list[Path] = []

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        slug = _url_slug(url)
        filename = f"{i:02d}-{slug}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  [OK] Saved: {filepath.name} ({filepath.stat().st_size:,} bytes)")
        saved.append(filepath)

    return saved


def crawl_all_sync(clean_first: bool = True) -> list[Path]:
    """Sync wrapper for pipeline runner."""
    return asyncio.run(crawl_all(clean_first=clean_first))


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("[WARN] Hay dien ARTICLE_URLS truoc khi chay!")
    else:
        files = crawl_all_sync()
        print(f"\nTong cong: {len(files)} bai bao trong {DATA_DIR}")
