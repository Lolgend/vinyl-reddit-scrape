"""Vinyl Releases Reddit Scraper."""

from reddit_scrape.db import (
    Base,
    get_database_url,
    get_engine,
    get_session_factory,
    init_db,
)
from reddit_scrape.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractedRelease,
    ExtractionResult,
    ExtractionWorker,
    create_extraction_agent,
    extract_releases,
    pre_clean_title,
)
from reddit_scrape.ingestion import (
    NON_RELEASE_FLAIRS,
    IngestionStats,
    RawSubmission,
    RedditSourceAdapter,
    SubmissionIngester,
    should_ingest_submission,
)
from reddit_scrape.models import (
    Artist,
    Release,
    ReleaseArtist,
    Submission,
    SubmissionRelease,
    SubmissionSnapshot,
)
from reddit_scrape.retention import (
    get_default_retention_days,
    purge_snapshots,
)

__all__ = [
    # Database
    "Base",
    "get_database_url",
    "get_engine",
    "get_session_factory",
    "init_db",
    # Models
    "Artist",
    "Release",
    "ReleaseArtist",
    "Submission",
    "SubmissionRelease",
    "SubmissionSnapshot",
    # Ingestion
    "NON_RELEASE_FLAIRS",
    "IngestionStats",
    "RawSubmission",
    "RedditSourceAdapter",
    "SubmissionIngester",
    "should_ingest_submission",
    # Extraction
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractedRelease",
    "ExtractionResult",
    "ExtractionWorker",
    "create_extraction_agent",
    "extract_releases",
    "pre_clean_title",
    # Retention
    "get_default_retention_days",
    "purge_snapshots",
]


def main() -> None:
    from reddit_scrape.cli import main as cli_main
    cli_main()
