"""Vinyl Releases Reddit Scraper."""

from reddit_scrape.db import (
    Base,
    get_database_url,
    get_engine,
    get_session_factory,
    init_db,
)
from reddit_scrape.extraction import (
    BATCH_EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    BatchExtractedItem,
    BatchExtractionResult,
    ExtractedRelease,
    ExtractionResult,
    ExtractionWorker,
    build_batch_extraction_prompt,
    build_extraction_prompt,
    create_batch_extraction_agent,
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
from reddit_scrape.persistence import (
    STANDARD_TOKEN,
    ReleasePersistenceService,
    compute_release_fingerprint,
    normalize_attribute,
    persist_extracted_releases,
)
from reddit_scrape.retention import (
    get_default_retention_days,
    purge_snapshots,
)
from reddit_scrape.service import (
    VALID_THINKING_LEVELS,
    CycleStats,
    PollerService,
    ServiceConfig,
    validate_environment,
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
    "BATCH_EXTRACTION_SYSTEM_PROMPT",
    "BatchExtractedItem",
    "BatchExtractionResult",
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractedRelease",
    "ExtractionResult",
    "ExtractionWorker",
    "build_batch_extraction_prompt",
    "build_extraction_prompt",
    "create_batch_extraction_agent",
    "create_extraction_agent",
    "extract_releases",
    "pre_clean_title",
    # Persistence
    "STANDARD_TOKEN",
    "ReleasePersistenceService",
    "compute_release_fingerprint",
    "normalize_attribute",
    "persist_extracted_releases",
    # Retention
    "get_default_retention_days",
    "purge_snapshots",
    # Service
    "CycleStats",
    "PollerService",
    "ServiceConfig",
    "VALID_THINKING_LEVELS",
    "validate_environment",
]


def main() -> None:
    from reddit_scrape.cli import main as cli_main

    cli_main()
