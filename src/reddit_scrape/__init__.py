"""Vinyl Releases Reddit Scraper."""

from reddit_scrape.db import (
    Base,
    get_database_url,
    get_engine,
    get_session_factory,
    init_db,
    reset_db,
)
from reddit_scrape.discogs import (
    DEFAULT_DISCOGS_BASE_URL,
    DEFAULT_DISCOGS_USER_AGENT,
    DEFAULT_RATE_LIMIT_FLOOR_SECONDS,
    DiscogsAPIError,
    DiscogsArtistCredit,
    DiscogsArtistDetail,
    DiscogsClient,
    DiscogsMasterDetail,
    DiscogsReleaseDetail,
    DiscogsSearchCandidate,
)
from reddit_scrape.embeddings import (
    compute_album_text,
    compute_artist_text,
    compute_release_text,
    embed_texts,
)
from reddit_scrape.extraction import (
    BATCH_EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    BatchExtractedItem,
    BatchExtractionResult,
    CandidateRetriever,
    ExtractedRelease,
    ExtractionResult,
    ExtractionWorker,
    build_batch_extraction_prompt,
    build_extraction_prompt,
    create_extraction_agent,
    extract_releases,
    format_catalog_reference,
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
    Album,
    AlbumArtist,
    Artist,
    Release,
    ReleaseArtist,
    Submission,
    SubmissionRelease,
    SubmissionSnapshot,
)
from reddit_scrape.persistence import (
    STANDARD_TOKEN,
    CatalogCandidates,
    CatalogReleaseCandidate,
    ReleasePersistenceService,
    compute_album_fingerprint,
    compute_release_fingerprint,
    find_catalog_candidates,
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

__all__ = [  # noqa: RUF022
    # Database
    "Base",
    "get_database_url",
    "get_engine",
    "get_session_factory",
    "init_db",
    "reset_db",
    # Models
    "Album",
    "AlbumArtist",
    "Artist",
    "Release",
    "ReleaseArtist",
    "Submission",
    "SubmissionRelease",
    "SubmissionSnapshot",
    # Embeddings
    "compute_album_text",
    "compute_artist_text",
    "compute_release_text",
    "embed_texts",
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
    "CandidateRetriever",
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractedRelease",
    "ExtractionResult",
    "ExtractionWorker",
    "build_batch_extraction_prompt",
    "build_extraction_prompt",
    "create_extraction_agent",
    "extract_releases",
    "format_catalog_reference",
    "pre_clean_title",
    # Persistence
    "CatalogCandidates",
    "CatalogReleaseCandidate",
    "STANDARD_TOKEN",
    "ReleasePersistenceService",
    "compute_album_fingerprint",
    "compute_release_fingerprint",
    "find_catalog_candidates",
    "normalize_attribute",
    "persist_extracted_releases",
    # Retention
    "get_default_retention_days",
    "purge_snapshots",
    # Discogs
    "DEFAULT_DISCOGS_BASE_URL",
    "DEFAULT_DISCOGS_USER_AGENT",
    "DEFAULT_RATE_LIMIT_FLOOR_SECONDS",
    "DiscogsAPIError",
    "DiscogsArtistCredit",
    "DiscogsArtistDetail",
    "DiscogsClient",
    "DiscogsMasterDetail",
    "DiscogsReleaseDetail",
    "DiscogsSearchCandidate",
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
