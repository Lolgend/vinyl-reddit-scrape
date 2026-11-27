import sys
import uuid

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)


class EnrichmentState(StrEnum):
    PENDING = "pending"
    MATCHED = "matched"
    PARTIAL_MATCHED = "partial_matched"
    NOT_FOUND = "not_found"
    FAILED = "failed"


class PipelineProcessState(StrEnum):
    """Operational lifecycle state of the background Pipeline process."""

    RUNNING = "running"
    STOPPED = "stopped"


class PipelineRunMode(StrEnum):
    """Execution mode of a Pipeline process."""

    CONTINUOUS = "continuous"
    ONCE = "once"


class PipelineLoopName(StrEnum):
    """Canonical identifier for a background Pipeline loop."""

    DISCOVERY_POLL = "discovery_poll"
    TRACKING_POLL = "tracking_poll"
    EXTRACTION_WORKER = "extraction_worker"
    ENRICHMENT_WORKER = "enrichment_worker"


class PipelineLoopStatus(StrEnum):
    """Operational state of an individual Pipeline loop."""

    RUNNING = "running"
    SLEEPING = "sleeping"
    ERROR = "error"
    STOPPED = "stopped"
    DISABLED = "disabled"


class PipelineRunStatus(StrEnum):
    """Terminal outcome status of a PipelineRun record."""

    SUCCESS = "success"
    ERROR = "error"


from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from reddit_scrape.db import Base


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(String(32), primary_key=True)
    reddit_id = Column(String(16), nullable=False, unique=True, index=True)
    title = Column(Text, nullable=False)
    url = Column(Text, nullable=False)
    flair = Column(Text, nullable=True)
    latest_score = Column(Integer, nullable=False, default=0)
    latest_upvote_ratio = Column(Float, nullable=False, default=1.0)
    latest_num_comments = Column(Integer, nullable=False, default=0)
    created_utc = Column(DateTime(timezone=True), nullable=False)
    is_extracted = Column(Boolean, nullable=False, default=False)
    is_release = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_submissions_created_utc", created_utc.desc()),
        Index(
            "idx_submissions_is_extracted",
            is_extracted,
            postgresql_where=text("is_extracted = FALSE"),
        ),
    )


class SubmissionSnapshot(Base):
    __tablename__ = "submission_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    submission_id = Column(
        String(32),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
    )
    score = Column(Integer, nullable=False)
    upvote_ratio = Column(Float, nullable=False)
    num_comments = Column(Integer, nullable=False)
    captured_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_snapshots_sub_time", submission_id, captured_at.desc()),
        Index("idx_snapshots_captured_at", captured_at),
    )


class Artist(Base):
    __tablename__ = "artists"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    name = Column(String(255), nullable=False, unique=True, index=True)
    discogs_artist_id = Column(Integer, nullable=True, unique=True)
    embedding = Column(Vector(384), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_artists_name", name),
        Index("idx_artists_discogs_artist_id", discogs_artist_id, unique=True),
        Index(
            "idx_artists_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Album(Base):
    __tablename__ = "albums"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    combined_artist = Column(String(255), nullable=False)
    title = Column(String(255), nullable=False)
    fingerprint = Column(String(64), nullable=False, unique=True)
    discogs_master_id = Column(Integer, nullable=True, unique=True)
    discogs_release_id = Column(Integer, nullable=True, unique=True)
    discogs_status = Column(String(32), nullable=False, default="pending", server_default=text("'pending'"))
    enrichment_attempts = Column(Integer, nullable=False, default=0, server_default=text("0"))
    last_enriched_at = Column(DateTime(timezone=True), nullable=True)
    enrichment_error = Column(Text, nullable=True)
    embedding = Column(Vector(384), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_albums_combined_artist", combined_artist),
        Index("idx_albums_title", title),
        Index("idx_albums_fingerprint", fingerprint, unique=True),
        Index("idx_albums_discogs_master_id", discogs_master_id, unique=True),
        Index("idx_albums_discogs_release_id", discogs_release_id, unique=True),
        Index(
            "idx_albums_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "idx_albums_enrichment_queue",
            discogs_status,
            last_enriched_at,
            created_at,
            postgresql_where=text("discogs_status IN ('pending', 'partial_matched', 'failed')"),
        ),
    )


class AlbumArtist(Base):
    __tablename__ = "album_artists"

    album_id = Column(
        UUID(as_uuid=True),
        ForeignKey("albums.id", ondelete="CASCADE"),
        primary_key=True,
    )
    artist_id = Column(
        UUID(as_uuid=True),
        ForeignKey("artists.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# Legacy alias
ReleaseArtist = AlbumArtist


class Release(Base):
    __tablename__ = "releases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    album_id = Column(
        UUID(as_uuid=True),
        ForeignKey("albums.id", ondelete="CASCADE"),
        nullable=False,
    )
    variant = Column(Text, nullable=True)
    retailer_name = Column(String(255), nullable=True)
    fingerprint = Column(String(64), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    album = relationship("Album", lazy="joined")

    @property
    def combined_artist(self) -> str:
        return self.album.combined_artist if self.album else ""

    @property
    def album_title(self) -> str:
        return self.album.title if self.album else ""

    @property
    def embedding(self) -> list[float] | None:
        return self.album.embedding if self.album else None

    __table_args__ = (
        Index("idx_releases_album_id", album_id),
        Index("idx_releases_fingerprint", fingerprint, unique=True),
    )


class SubmissionRelease(Base):
    __tablename__ = "submission_releases"

    submission_id = Column(
        String(32),
        ForeignKey("submissions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    release_id = Column(
        UUID(as_uuid=True),
        ForeignKey("releases.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PipelineProcessStatus(Base):
    """Singleton operational status row for the active Pipeline worker process."""

    __tablename__ = "pipeline_process_status"

    process_name = Column(String(64), primary_key=True, default="default")
    status = Column(String(32), nullable=False)
    run_mode = Column(String(16), nullable=False)
    host = Column(String(255), nullable=False)
    pid = Column(Integer, nullable=False)
    heartbeat_interval_seconds = Column(Integer, nullable=False, default=30)
    started_at = Column(DateTime(timezone=True), nullable=False)
    last_heartbeat_at = Column(DateTime(timezone=True), nullable=False)
    stopped_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class PipelineLoopState(Base):
    """Latest operational state and schedule countdown for each Pipeline loop."""

    __tablename__ = "pipeline_loop_states"

    loop_name = Column(String(32), primary_key=True)
    status = Column(String(32), nullable=False)
    interval_seconds = Column(Integer, nullable=False)
    last_started_at = Column(DateTime(timezone=True), nullable=True)
    last_completed_at = Column(DateTime(timezone=True), nullable=True)
    next_run_at = Column(DateTime(timezone=True), nullable=True)
    last_duration_ms = Column(Integer, nullable=True)
    last_run_stats = Column(JSONB, nullable=True)
    last_error = Column(Text, nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class PipelineRun(Base):
    """Historical execution log entry for a Pipeline loop cycle."""

    __tablename__ = "pipeline_runs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    loop_name = Column(String(32), nullable=False)
    status = Column(String(16), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=False)
    duration_ms = Column(Integer, nullable=False)
    stats = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    host = Column(String(255), nullable=True)
    pid = Column(Integer, nullable=True)

    __table_args__ = (
        Index("idx_pipeline_runs_started_at", started_at.desc()),
        Index("idx_pipeline_runs_loop_started", loop_name, started_at.desc()),
        Index("idx_pipeline_runs_status_started", status, started_at.desc()),
    )


__all__ = [
    "Album",
    "AlbumArtist",
    "Artist",
    "EnrichmentState",
    "PipelineLoopName",
    "PipelineLoopState",
    "PipelineLoopStatus",
    "PipelineProcessState",
    "PipelineProcessStatus",
    "PipelineRun",
    "PipelineRunMode",
    "PipelineRunStatus",
    "Release",
    "ReleaseArtist",
    "Submission",
    "SubmissionRelease",
    "SubmissionSnapshot",
]
