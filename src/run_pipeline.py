"""
Chay toan bo pipeline: Task 1 -> 2 -> 3 -> 4.

Usage:
    python -m src.run_pipeline
    python -m src.run_pipeline --skip-crawl   # chi convert + index
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"


def _clean_stale_standardized() -> None:
    """Xoa markdown cu khong con file landing tuong ung."""
    landing_legal = {p.stem for p in (DATA_DIR / "landing" / "legal").glob("*.*")}
    landing_news = {p.stem for p in (DATA_DIR / "landing" / "news").glob("*.json")}

    for md in (DATA_DIR / "standardized" / "legal").glob("*.md"):
        if md.stem not in landing_legal:
            md.unlink()
            print(f"[CLEAN] Removed stale: {md.name}")

    for md in (DATA_DIR / "standardized" / "news").glob("*.md"):
        if md.stem not in landing_news:
            md.unlink()
            print(f"[CLEAN] Removed stale: {md.name}")


def run(skip_crawl: bool = False, reindex: bool = True) -> None:
    if not skip_crawl:
        from src.task1_collect_legal_docs import download_all
        from src.task2_crawl_news import crawl_all_sync

        print("\n=== TASK 1: Collect legal documents ===")
        download_all(skip_existing=False)

        print("\n=== TASK 2: Crawl news articles ===")
        crawl_all_sync()

    print("\n=== TASK 3: Convert to Markdown ===")
    _clean_stale_standardized()
    from src.task3_convert_markdown import convert_all

    convert_all()

    if reindex:
        print("\n=== TASK 4: Chunking & Indexing ===")
        from src.task4_chunking_indexing import run_pipeline

        run_pipeline()

    print("\n[OK] Pipeline hoan tat!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--no-reindex", action="store_true")
    args = parser.parse_args()
    run(skip_crawl=args.skip_crawl, reindex=not args.no_reindex)
