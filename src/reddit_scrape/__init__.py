"""Vinyl Releases Reddit Scraper."""

from reddit_scrape.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractedRelease,
    ExtractionResult,
    ExtractionWorker,
    create_extraction_agent,
    extract_releases,
    pre_clean_title,
)

__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractedRelease",
    "ExtractionResult",
    "ExtractionWorker",
    "create_extraction_agent",
    "extract_releases",
    "pre_clean_title",
]


def main() -> None:
    print("Hello from reddit-scrape!")
