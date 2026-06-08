"""Shared Crawl4AI configuration for Task 1 and Task 2."""

from crawl4ai import CrawlerRunConfig

# domcontentloaded: on dinh hon networkidle voi cac trang VN co nhieu tracker/quang cao
DEFAULT_CRAWL_CONFIG = CrawlerRunConfig(
    page_timeout=120_000,
    wait_until="domcontentloaded",
)
