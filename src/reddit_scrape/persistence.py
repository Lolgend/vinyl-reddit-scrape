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
    compute_album_text,
    compute_artist_text,
    embed_texts,
)
from reddit_scrape.extraction import (
    ExtractedRelease,
    ExtractionResult,
    pre_clean_title,
)
from reddit_scrape.models import (
    Album,
    AlbumArtist,
    Artist,
    Release,
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
    album_fingerprint: str
    album_text: str
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


def compute_album_fingerprint(
    combined_artist: str,
    album_title: str,
) -> str:
    """Compute a deterministic SHA-256 fingerprint over normalized album metadata.

    Format: sha256("{norm_combined_artist}|{norm_title}")
    """
    norm_artist = normalize_attribute(combined_artist)
    norm_album = normalize_attribute(album_title)
    payload = f"{norm_artist}|{norm_album}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_release_fingerprint(
    album_id: Any,
    variant: str | None = None,
    retailer: str | None = None,
) -> str:
    """Compute a deterministic SHA-256 fingerprint over release metadata scoped to an album.

    Format: sha256("{album_id}|{norm_variant}|{norm_retailer}")
    Missing variant or retailer maps to "[standard]".
    """
    norm_album_id = str(album_id).strip().lower()
    norm_variant = normalize_attribute(variant, default_token=STANDARD_TOKEN)
    norm_retailer = normalize_attribute(retailer, default_token=STANDARD_TOKEN)

    payload = f"{norm_album_id}|{norm_variant}|{norm_retailer}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def persist_extracted_releases(
    session: Session,
    submission_id: str,
    releases: ExtractionResult | Sequence[ExtractedRelease],
) -> list[Release]:
    """Persist extracted releases within the caller's active database session.

    Atomically:
    1. Generates 384-dimensional dense vector embeddings for albums and contributing artists.
    2. Idempotently upserts Album records (pending on insert, preserving existing embedding and discogs_status).
    3. Idempotently upserts Artist records and links them in album_artists.
    4. Idempotently upserts Release records scoped by album_id.
    5. Links parent submission to releases in submission_releases (ON CONFLICT DO NOTHING).
    6. Marks submission.is_extracted = TRUE.
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

        album_fp = compute_album_fingerprint(
            combined_artist=combined_artist,
            album_title=album_title,
        )

        album_text = compute_album_text(
            combined_artist=combined_artist,
            album_title=album_title,
        )
        ordered_texts.append(album_text)

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
                album_fingerprint=album_fp,
                album_text=album_text,
                artist_names=artist_names,
            )
        )

    unique_texts = list(dict.fromkeys(ordered_texts))
    embeddings_list = embed_texts(unique_texts)
    embedding_map = dict(zip(unique_texts, embeddings_list))

    persisted_releases: list[Release] = []

    for item in items_to_persist:
        album_embedding = embedding_map[item.album_text]

        # 1. Upsert Album (discogs_status='pending' on insert, preserved on conflict)
        album_stmt = pg_insert(Album).values(
            combined_artist=item.combined_artist,
            title=item.album_title,
            fingerprint=item.album_fingerprint,
            discogs_status="pending",
            enrichment_attempts=0,
            embedding=album_embedding,
        )
        album_upsert = album_stmt.on_conflict_do_update(
            index_elements=[Album.fingerprint],
            set_={
                "updated_at": func.now(),
                "embedding": func.coalesce(Album.embedding, album_stmt.excluded.embedding),
            },
        ).returning(Album.id)
        album_id = session.execute(album_upsert).scalar_one()

        # 2. Upsert constituent Artists and link AlbumArtist
        for art_name in item.artist_names:
            art_text = compute_artist_text(art_name)
            art_embedding = embedding_map[art_text]

            art_stmt = pg_insert(Artist).values(
                name=art_name,
                embedding=art_embedding,
            )
            art_upsert = art_stmt.on_conflict_do_update(
                index_elements=[Artist.name],
                set_={
                    "updated_at": func.now(),
                    "embedding": func.coalesce(Artist.embedding, art_stmt.excluded.embedding),
                },
            ).returning(Artist.id)
            art_id = session.execute(art_upsert).scalar_one()

            aa_stmt = (
                pg_insert(AlbumArtist)
                .values(album_id=album_id, artist_id=art_id)
                .on_conflict_do_nothing()
            )
            session.execute(aa_stmt)

        # 3. Upsert Release scoped by album_id
        rel_fp = compute_release_fingerprint(
            album_id=album_id,
            variant=item.variant,
            retailer=item.retailer,
        )
        rel_stmt = pg_insert(Release).values(
            album_id=album_id,
            variant=item.variant,
            retailer_name=item.retailer,
            fingerprint=rel_fp,
        )
        rel_upsert = rel_stmt.on_conflict_do_update(
            index_elements=[Release.fingerprint],
            set_={
                "updated_at": func.now(),
            },
        ).returning(Release.id)
        rel_id = session.execute(rel_upsert).scalar_one()

        # 4. Link SubmissionRelease
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
    model: type[Album | Artist],
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
    3. Queries PostgreSQL for up to `top_k` Albums and `top_k` Artists per query vector
       having cosine distance <= (1.0 - min_similarity).
    4. Deduplicates albums and artists across the query batch, keeping the highest similarity.
    5. Hydrates contributing artists from `AlbumArtist` -> `Artist`.
    6. Hydrates child `Release` variants and retailers for matched albums.
    7. Returns clean `CatalogCandidates` sorted by similarity descending.
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

    # Map album ID to (Album, best_similarity)
    best_albums: dict[Any, tuple[Album, float]] = {}
    # Map artist name to best_similarity
    best_artists: dict[str, float] = {}

    max_distance = 1.0 - min_similarity

    for title in valid_titles:
        vec = vector_map[title]

        # 1. Query top_k Albums using pgvector cosine distance
        album_stmt = _build_cosine_similarity_query(Album, vec, max_distance, top_k)
        album_rows = session.execute(album_stmt).all()
        for album_obj, dist in album_rows:
            sim = 1.0 - float(dist)
            if sim >= min_similarity and (
                album_obj.id not in best_albums or sim > best_albums[album_obj.id][1]
            ):
                best_albums[album_obj.id] = (album_obj, sim)

        # 2. Query top_k Artists using pgvector cosine distance
        art_stmt = _build_cosine_similarity_query(Artist, vec, max_distance, top_k)
        art_rows = session.execute(art_stmt).all()
        for art_obj, dist in art_rows:
            sim = 1.0 - float(dist)
            if sim >= min_similarity and (
                art_obj.name not in best_artists or sim > best_artists[art_obj.name]
            ):
                best_artists[art_obj.name] = sim

    # 3. Hydrate contributing artists for all matched albums
    matched_album_ids = list(best_albums.keys())
    album_artists_map: dict[Any, list[str]] = {aid: [] for aid in matched_album_ids}
    if matched_album_ids:
        aa_stmt = (
            select(AlbumArtist.album_id, Artist.name)
            .join(Artist, AlbumArtist.artist_id == Artist.id)
            .where(AlbumArtist.album_id.in_(matched_album_ids))
        )
        aa_rows = session.execute(aa_stmt).all()
        for a_id, a_name in aa_rows:
            if a_name not in album_artists_map[a_id]:
                album_artists_map[a_id].append(str(a_name))

    # 4. Hydrate child releases for matched albums
    album_releases_map: dict[Any, list[Release]] = {aid: [] for aid in matched_album_ids}
    if matched_album_ids:
        rel_stmt = (
            select(Release)
            .where(Release.album_id.in_(matched_album_ids))
            .order_by(Release.created_at.asc())
        )
        rel_rows = session.execute(rel_stmt).scalars().all()
        for r in rel_rows:
            album_releases_map[r.album_id].append(r)

    # 5. Assemble deduplicated CatalogReleaseCandidate list sorted by similarity descending
    sorted_albums = sorted(best_albums.values(), key=lambda pair: pair[1], reverse=True)
    candidate_releases: list[CatalogReleaseCandidate] = []
    seen_rel_ids: set[str] = set()

    for album_obj, sim in sorted_albums:
        artists = album_artists_map.get(album_obj.id, [])
        if not artists and album_obj.combined_artist:
            artists = [str(album_obj.combined_artist)]

        child_releases = album_releases_map.get(album_obj.id, [])
        if child_releases:
            for child in child_releases:
                cid = str(child.id)
                if cid not in seen_rel_ids:
                    seen_rel_ids.add(cid)
                    candidate_releases.append(
                        CatalogReleaseCandidate(
                            id=cid,
                            combined_artist=str(album_obj.combined_artist),
                            album_title=str(album_obj.title),
                            artists=artists,
                            variant=str(child.variant) if child.variant is not None else None,
                            retailer=str(child.retailer_name) if child.retailer_name is not None else None,
                            similarity=round(sim, 4),
                        )
                    )
        else:
            aid = str(album_obj.id)
            if aid not in seen_rel_ids:
                seen_rel_ids.add(aid)
                candidate_releases.append(
                    CatalogReleaseCandidate(
                        id=aid,
                        combined_artist=str(album_obj.combined_artist),
                        album_title=str(album_obj.title),
                        artists=artists,
                        variant=None,
                        retailer=None,
                        similarity=round(sim, 4),
                    )
                )

    # 6. Assemble deduplicated artist names sorted by similarity descending
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
    "compute_album_fingerprint",
    "compute_release_fingerprint",
    "embed_texts",
    "find_catalog_candidates",
    "normalize_attribute",
    "persist_extracted_releases",
]
