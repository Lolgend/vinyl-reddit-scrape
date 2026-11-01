"""Relational persistence and release deduplication engine."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from reddit_scrape.embeddings import (
    compute_artist_text,
    compute_release_text,
    embed_texts,
)
from reddit_scrape.extraction import (
    ExtractedRelease,
    ExtractionResult,
    pre_clean_title,
)
from reddit_scrape.models import (
    Artist,
    Release,
    ReleaseArtist,
    Submission,
    SubmissionRelease,
)

STANDARD_TOKEN = "[standard]"


@dataclass(frozen=True)
class CatalogReleaseCandidate:
    """Canonical release candidate retrieved from vector similarity search."""

    id: str
    combined_artist: str
    album_title: str
    artists: list[str]
    variant: str | None = None
    retailer: str | None = None
    similarity: float = 0.0


@dataclass(frozen=True)
class CatalogCandidates:
    """Catalog similarity match candidates containing releases and artists."""

    releases: list[CatalogReleaseCandidate] = field(default_factory=list)
    artists: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.releases and not self.artists

    def __bool__(self) -> bool:
        return bool(self.releases or self.artists)


@dataclass(frozen=True)
class _PreparedReleaseItem:
    """Normalized release metadata and embedding text prepared for persistence."""

    combined_artist: str
    album_title: str
    variant: str | None
    retailer: str | None
    fingerprint: str
    rel_text: str
    artist_names: list[str]


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
        sub.is_release = False  # type: ignore[assignment]
        sub.is_extracted = True  # type: ignore[assignment]
        sub.updated_at = func.now()  # type: ignore[assignment]
        session.flush()
        return []

    # Prepare metadata and collect texts for batch vector embedding in deterministic order
    items_to_persist: list[_PreparedReleaseItem] = []
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
            _PreparedReleaseItem(
                combined_artist=combined_artist,
                album_title=album_title,
                variant=variant,
                retailer=retailer,
                fingerprint=fingerprint,
                rel_text=rel_text,
                artist_names=artist_names,
            )
        )

    unique_texts = list(dict.fromkeys(ordered_texts))
    embeddings_list = embed_texts(unique_texts)
    embedding_map = dict(zip(unique_texts, embeddings_list))

    persisted_releases: list[Release] = []

    for item in items_to_persist:
        rel_embedding = embedding_map[item.rel_text]

        rel_stmt = pg_insert(Release).values(
            combined_artist=item.combined_artist,
            album_title=item.album_title,
            variant=item.variant,
            retailer_name=item.retailer,
            fingerprint=item.fingerprint,
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
        for art_name in item.artist_names:
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

    sub.is_extracted = True  # type: ignore[assignment]
    sub.updated_at = func.now()  # type: ignore[assignment]
    session.flush()

    return persisted_releases


def _build_cosine_similarity_query(
    model: type[Release | Artist],
    vector: list[float],
    max_distance: float,
    top_k: int,
) -> Select[Any]:
    """Build a cosine similarity query filtering by maximum distance and ordering nearest first."""
    dist_expr = model.embedding.cosine_distance(vector)
    return (
        select(model, dist_expr.label("dist"))
        .where(model.embedding.is_not(None))
        .where(dist_expr <= (max_distance + 1e-6))
        .order_by(dist_expr.asc())
        .limit(top_k)
    )


def find_catalog_candidates(
    session: Session,
    query_titles: str | Sequence[str],
    top_k: int = 5,
    min_similarity: float = 0.50,
) -> CatalogCandidates:
    """Find semantically similar catalog releases and artists using pgvector cosine distance.

    1. Normalizes input query strings and pre-cleans them with `pre_clean_title`.
    2. Embeds all unique query titles in a single vectorized batch call using `embed_texts`.
    3. Queries PostgreSQL for up to `top_k` Releases and `top_k` Artists per query vector
       having cosine distance <= (1.0 - min_similarity).
    4. Deduplicates releases and artists across the query batch, keeping the highest similarity.
    5. Hydrates matched releases with their contributing artists from `ReleaseArtist` / `Artist`.
    6. Returns clean `CatalogCandidates` sorted by similarity descending.
    """
    if isinstance(query_titles, str):
        raw_titles = [query_titles]
    else:
        raw_titles = list(query_titles)

    cleaned_titles = [pre_clean_title(t) for t in raw_titles]
    valid_titles = [t for t in cleaned_titles if t]
    if not valid_titles:
        return CatalogCandidates()

    # Deduplicate titles for single batch embedding pass
    unique_titles = list(dict.fromkeys(valid_titles))
    vectors = embed_texts(unique_titles)
    vector_map = dict(zip(unique_titles, vectors))

    # Map release ID to (Release, best_similarity)
    best_releases: dict[Any, tuple[Release, float]] = {}
    # Map artist name to best_similarity
    best_artists: dict[str, float] = {}

    max_distance = 1.0 - min_similarity

    for title in valid_titles:
        vec = vector_map[title]

        # 1. Query top_k Releases using pgvector cosine distance
        rel_stmt = _build_cosine_similarity_query(Release, vec, max_distance, top_k)
        rel_rows = session.execute(rel_stmt).all()
        for rel_obj, dist in rel_rows:
            sim = 1.0 - float(dist)
            if sim >= min_similarity and (
                rel_obj.id not in best_releases or sim > best_releases[rel_obj.id][1]
            ):
                best_releases[rel_obj.id] = (rel_obj, sim)

        # 2. Query top_k Artists using pgvector cosine distance
        art_stmt = _build_cosine_similarity_query(Artist, vec, max_distance, top_k)
        art_rows = session.execute(art_stmt).all()
        for art_obj, dist in art_rows:
            sim = 1.0 - float(dist)
            if sim >= min_similarity and (
                art_obj.name not in best_artists or sim > best_artists[art_obj.name]
            ):
                best_artists[art_obj.name] = sim

    # 3. Hydrate contributing artists for all matched releases
    matched_rel_ids = list(best_releases.keys())
    rel_artists_map: dict[Any, list[str]] = {rid: [] for rid in matched_rel_ids}
    if matched_rel_ids:
        ra_stmt = (
            select(ReleaseArtist.release_id, Artist.name)
            .join(Artist, ReleaseArtist.artist_id == Artist.id)
            .where(ReleaseArtist.release_id.in_(matched_rel_ids))
        )
        ra_rows = session.execute(ra_stmt).all()
        for r_id, a_name in ra_rows:
            if a_name not in rel_artists_map[r_id]:
                rel_artists_map[r_id].append(str(a_name))

    # 4. Assemble deduplicated CatalogReleaseCandidate list sorted by similarity descending
    sorted_releases = sorted(best_releases.values(), key=lambda pair: pair[1], reverse=True)
    candidate_releases: list[CatalogReleaseCandidate] = []
    for rel_obj, sim in sorted_releases:
        artists = rel_artists_map.get(rel_obj.id, [])
        if not artists and rel_obj.combined_artist:
            artists = [str(rel_obj.combined_artist)]
        candidate_releases.append(
            CatalogReleaseCandidate(
                id=str(rel_obj.id),
                combined_artist=str(rel_obj.combined_artist),
                album_title=str(rel_obj.album_title),
                artists=artists,
                variant=str(rel_obj.variant) if rel_obj.variant is not None else None,
                retailer=str(rel_obj.retailer_name) if rel_obj.retailer_name is not None else None,
                similarity=round(sim, 4),
            )
        )

    # 5. Assemble deduplicated artist names sorted by similarity descending
    sorted_artists = sorted(best_artists.items(), key=lambda pair: pair[1], reverse=True)
    candidate_artists = [str(name) for name, _ in sorted_artists]

    return CatalogCandidates(releases=candidate_releases, artists=candidate_artists)


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

    def find_candidates(
        self,
        query_titles: str | Sequence[str],
        top_k: int = 5,
        min_similarity: float = 0.50,
    ) -> CatalogCandidates:
        """Find semantically similar catalog releases and artists for query titles."""
        with self.session_factory() as session:
            return find_catalog_candidates(
                session=session,
                query_titles=query_titles,
                top_k=top_k,
                min_similarity=min_similarity,
            )


__all__ = [
    "STANDARD_TOKEN",
    "CatalogCandidates",
    "CatalogReleaseCandidate",
    "ReleasePersistenceService",
    "compute_release_fingerprint",
    "embed_texts",
    "find_catalog_candidates",
    "normalize_attribute",
    "persist_extracted_releases",
]
