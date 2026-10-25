import uuid

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
from sqlalchemy.dialects.postgresql import UUID

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
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_artists_name", name),
    )


class Release(Base):
    __tablename__ = "releases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=text("gen_random_uuid()"))
    combined_artist = Column(String(255), nullable=False, index=True)
    album_title = Column(String(255), nullable=False, index=True)
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

    __table_args__ = (
        Index("idx_releases_combined_artist", combined_artist),
        Index("idx_releases_album", album_title),
        Index("idx_releases_fingerprint", fingerprint, unique=True),
    )


class ReleaseArtist(Base):
    __tablename__ = "release_artists"

    release_id = Column(
        UUID(as_uuid=True),
        ForeignKey("releases.id", ondelete="CASCADE"),
        primary_key=True,
    )
    artist_id = Column(
        UUID(as_uuid=True),
        ForeignKey("artists.id", ondelete="CASCADE"),
        primary_key=True,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


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
