"""Relational persistence and release deduplication engine."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from reddit_scrape.embeddings import (
    compute_artist_text,
    compute_release_text,
    embed_texts,
)
from reddit_scrape.extraction import ExtractedRelease, ExtractionResult
from reddit_scrape.models import (
    Artist,
    Release,
    ReleaseArtist,
    Submission,
    SubmissionRelease,
)

STANDARD_TOKEN = "[standard]"


def normalize_attribute(val: str | None, default_token: str | None = None) -> str:
    """Normalize release metadata attribute for deterministic fingerprinting.

    Lowercases, collapses whitespace, normalizes unicode dashes to standard hyphens,
    and falls back to default_token if empty or missing.
    """
    if val is None:
        return default_token if default_token is not None else ""

    raw = str(val).strip()
    if not raw:
        return default_token if default_token is not None else ""

    # If already a standard placeholder token
    if raw.lower() in (STANDARD_TOKEN, "standard") and default_token == STANDARD_TOKEN:
        return STANDARD_TOKEN

    # Normalize unicode dashes (\u2010-\u2015 and \u2212) to standard ASCII hyphen
    cleaned = re.sub(r"[\u2010-\u2015\u2212]", "-", raw)

    # Collapse multiple whitespaces and lowercase
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()

    if not cleaned:
        return default_token if default_token is not None else ""

    return cleaned


def compute_release_fingerprint(
    combined_artist: str,
    album_title: str,
    variant: str | None = None,
    retailer: str | None = None,
) -> str:
    """Compute a deterministic SHA-256 fingerprint over normalized release metadata.

    Format: sha256("{norm_artist}|{norm_album}|{norm_variant}|{norm_retailer}")
    Missing variant or retailer maps to "[standard]".
    """
    norm_artist = normalize_attribute(combined_artist)
    norm_album = normalize_attribute(album_title)
    norm_variant = normalize_attribute(variant, default_token=STANDARD_TOKEN)
    norm_retailer = normalize_attribute(retailer, default_token=STANDARD_TOKEN)

    payload = f"{norm_artist}|{norm_album}|{norm_variant}|{norm_retailer}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def persist_extracted_releases(
    session: Session,
    submission_id: str,
    releases: ExtractionResult | Sequence[ExtractedRelease],
) -> list[Release]:
    """Persist extracted releases within the caller's active database session.

    Atomically:
    1. Generates 384-dimensional dense vector embeddings for releases and contributing artists.
    2. Idempotently upserts Release records based on fingerprint conflict, preserving/populating embeddings.
    3. Idempotently inserts/updates Artist records with embeddings and links them in release_artists.
    4. Links parent submission to releases in submission_releases (ON CONFLICT DO NOTHING).
    5. Marks submission.is_extracted = TRUE.
    """
    sub = session.get(Submission, submission_id)
    if sub is None:
        alt_id = submission_id[3:] if submission_id.startswith("t3_") else f"t3_{submission_id}"
        sub = session.get(Submission, alt_id)
        if sub is None:
            sub = session.execute(
                select(Submission).where(
                    (Submission.reddit_id == submission_id)
                    | (Submission.reddit_id == alt_id)
                )
            ).scalars().first()

    if sub is None:
        raise ValueError(f"Submission with ID '{submission_id}' does not exist.")

    if isinstance(releases, ExtractionResult):
        release_items = releases.releases
    else:
        release_items = list(releases)

    if not release_items:
        sub.is_release = False
        sub.is_extracted = True
        sub.updated_at = func.now()
        session.flush()
        return []

    # Prepare metadata and collect texts for batch vector embedding in deterministic order
    items_to_persist = []
    ordered_texts: list[str] = []

    for ext in release_items:
        combined_artist = (
            ext.combined_artist.strip()
            if ext.combined_artist and ext.combined_artist.strip()
            else (" / ".join(ext.artists).strip() if ext.artists else "")
        )
        album_title = ext.album_title.strip() if ext.album_title else ""
        variant = ext.variant.strip() if ext.variant and ext.variant.strip() else None
        retailer = ext.retailer.strip() if ext.retailer and ext.retailer.strip() else None

        fingerprint = compute_release_fingerprint(
            combined_artist=combined_artist,
            album_title=album_title,
            variant=variant,
            retailer=retailer,
        )

        rel_text = compute_release_text(
            combined_artist=combined_artist,
            album_title=album_title,
            variant=variant,
            retailer=retailer,
        )
        ordered_texts.append(rel_text)

        contributing_artists = ext.artists if ext.artists else ([combined_artist] if combined_artist else [])
        artist_names = list(
            dict.fromkeys(a.strip() for a in contributing_artists if a.strip())
        )
        for art_name in artist_names:
            ordered_texts.append(compute_artist_text(art_name))

        items_to_persist.append(
            (combined_artist, album_title, variant, retailer, fingerprint, rel_text, artist_names)
        )

    unique_texts = list(dict.fromkeys(ordered_texts))
    embeddings_list = embed_texts(unique_texts)
    embedding_map = dict(zip(unique_texts, embeddings_list))

    persisted_releases: list[Release] = []

    for (
        combined_artist,
        album_title,
        variant,
        retailer,
        fingerprint,
        rel_text,
        artist_names,
    ) in items_to_persist:
        rel_embedding = embedding_map[rel_text]

        rel_stmt = pg_insert(Release).values(
            combined_artist=combined_artist,
            album_title=album_title,
            variant=variant,
            retailer_name=retailer,
            fingerprint=fingerprint,
            embedding=rel_embedding,
        )
        rel_returning = rel_stmt.on_conflict_do_update(
            index_elements=[Release.fingerprint],
            set_={
                "updated_at": func.now(),
                "embedding": func.coalesce(Release.embedding, rel_stmt.excluded.embedding),
            },
        ).returning(Release.id)
        rel_id = session.execute(rel_returning).scalar_one()

        # Link contributing artists
        for art_name in artist_names:
            art_text = compute_artist_text(art_name)
            art_embedding = embedding_map[art_text]

            art_stmt = pg_insert(Artist).values(
                name=art_name,
                embedding=art_embedding,
            )
            art_returning = art_stmt.on_conflict_do_update(
                index_elements=[Artist.name],
                set_={
                    "name": art_stmt.excluded.name,
                    "embedding": func.coalesce(Artist.embedding, art_stmt.excluded.embedding),
                },
            ).returning(Artist.id)
            art_id = session.execute(art_returning).scalar_one()

            ra_stmt = (
                pg_insert(ReleaseArtist)
                .values(release_id=rel_id, artist_id=art_id)
                .on_conflict_do_nothing()
            )
            session.execute(ra_stmt)

        # Link submission to release
        sr_stmt = (
            pg_insert(SubmissionRelease)
            .values(submission_id=sub.id, release_id=rel_id)
            .on_conflict_do_nothing()
        )
        session.execute(sr_stmt)

        rel_obj = session.get(Release, rel_id)
        if rel_obj:
            persisted_releases.append(rel_obj)

    sub.is_extracted = True
    sub.updated_at = func.now()
    session.flush()

    return persisted_releases


class ReleasePersistenceService:
    """Service managing database transactions for release persistence and deduplication."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory

    def persist(
        self,
        submission_id: str,
        releases: ExtractionResult | Sequence[ExtractedRelease],
    ) -> list[Release]:
        """Atomically persist extracted releases for a submission in a managed transaction."""
        with self.session_factory() as session, session.begin():
            result = persist_extracted_releases(
                session=session,
                submission_id=submission_id,
                releases=releases,
            )
            for r in result:
                session.refresh(r)
            session.expunge_all()
            return result


__all__ = [
    "STANDARD_TOKEN",
    "ReleasePersistenceService",
    "compute_release_fingerprint",
    "embed_texts",
    "normalize_attribute",
    "persist_extracted_releases",
]
