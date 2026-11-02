"""Discogs catalog enrichment, multi-tier search cascade, and LLM candidate arbitration."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from reddit_scrape.discogs import DiscogsClient, DiscogsSearchCandidate
from reddit_scrape.embeddings import (
    compute_album_text,
    compute_artist_text,
    embed_texts,
)
from reddit_scrape.extraction import get_default_thinking_level

load_dotenv()

logger = logging.getLogger(__name__)


CANDIDATE_ARBITER_SYSTEM_PROMPT = """You are an expert discography classifier and catalog reconciliation arbiter.
Your role is to evaluate Discogs database search candidates against a target musical work or artist entity.

Evaluation Rules:
1. Musical Work Matching (Tiers 1-4):
   - Determine if any candidate represents the exact same musical work (album, EP, single, split) as the target.
   - Punctuation differences, reversed split artist order (e.g. 'Dropdead / Converge' vs 'Converge / Dropdead'), and subtitle variations do not disqualify a match if the underlying musical work is identical.
   - Do NOT match tribute albums, live bootlegs (unless the target is that specific live bootleg), or unrelated works sharing common words.
   - Discogs 'release' candidates represent specific physical pressings of the work, whereas 'master' candidates represent the parent work. Either can be a valid work match.

2. Artist Matching (Tier 5):
   - Determine if any candidate represents the exact same musical artist, performer, or band.
   - Discogs appends numeric homonym suffixes to distinguish different artists with the same name (e.g. 'Nirvana (2)', 'The Smile (5)'). Match the candidate if it represents the intended musical artist.

3. Output Requirements:
   - If a candidate matches, return `is_match=True` and `matched_candidate_id` set to the integer ID of the matching candidate.
   - If none of the candidates match, return `is_match=False` and `matched_candidate_id=None`.
   - Provide concise `reasoning` explaining your decision.
"""


class CandidateArbiterResult(BaseModel):
    """Structured LLM evaluation of Discogs search candidates."""

    is_match: bool = Field(
        description="True if one of the candidates represents the exact same musical work or artist."
    )
    matched_candidate_id: int | None = Field(
        None,
        description="The Discogs ID of the winning candidate, or None if no candidate matches.",
    )
    confidence: float | None = Field(
        None,
        description="Confidence score between 0.0 and 1.0.",
    )
    reasoning: str | None = Field(
        None,
        description="Brief explanation of why the candidate was confirmed or why no candidates matched.",
    )


@dataclass
class ResolvedWorkMatch:
    """A confirmed musical work match from Tiers 1-4 with resolved master/release IDs."""

    tier: int
    candidate: DiscogsSearchCandidate
    resolved_master_id: int | None
    resolved_release_id: int | None
    cosine_similarity: float
    search_rank: int
    reasoning: str | None = None


@dataclass
class ResolvedArtistMatch:
    """A confirmed constituent artist match from Tier 5."""

    artist_name: str
    candidate: DiscogsSearchCandidate
    discogs_artist_id: int
    cosine_similarity: float
    search_rank: int
    reasoning: str | None = None


@dataclass
class ConstituentArtistInfo:
    """Constituent artist name and optional existing discogs_artist_id."""

    name: str
    discogs_artist_id: int | None = None


@dataclass
class CascadeResult:
    """Outcome of the 5-tier Discogs search cascade."""

    work_match: ResolvedWorkMatch | None = None
    artist_matches: dict[str, ResolvedArtistMatch] = field(default_factory=dict)
    tier_reached: int = 1
    tiers_executed: list[int] = field(default_factory=list)


@dataclass
class ScoredCandidate:
    """A Discogs search candidate annotated with fastembed cosine similarity and search rank."""

    candidate: DiscogsSearchCandidate
    cosine_similarity: float
    search_rank: int

    @property
    def id(self) -> int:
        return self.candidate.id

    @property
    def type(self) -> str:
        return self.candidate.type

    @property
    def title(self) -> str:
        return self.candidate.title

    @property
    def master_id(self) -> int | None:
        return self.candidate.master_id

    @property
    def year(self) -> str | int | None:
        return self.candidate.year

    @property
    def country(self) -> str | None:
        return self.candidate.country

    @property
    def format(self) -> list[str]:
        return self.candidate.format


def compute_cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def deduplicate_candidates(
    candidates: Sequence[DiscogsSearchCandidate],
    max_candidates: int = 5,
) -> list[DiscogsSearchCandidate]:
    """Deduplicate candidates by (type, id) preserving order and capping at max_candidates."""
    seen: set[tuple[str, int]] = set()
    deduped: list[DiscogsSearchCandidate] = []
    for c in candidates:
        key = (c.type, c.id)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
            if len(deduped) >= max_candidates:
                break
    return deduped


def score_candidates(
    query_text: str,
    candidates: Sequence[DiscogsSearchCandidate],
    max_candidates: int = 5,
) -> list[ScoredCandidate]:
    """Deduplicate candidates and compute fastembed cosine similarity against query_text.

    All candidates (up to max_candidates) are preserved without any hard cosine cutoff filter.
    """
    deduped = deduplicate_candidates(candidates, max_candidates=max_candidates)
    if not deduped:
        return []

    texts_to_embed = [query_text] + [c.title for c in deduped]
    vectors = embed_texts(texts_to_embed)
    query_vec = vectors[0]
    candidate_vecs = vectors[1:]

    scored: list[ScoredCandidate] = []
    for rank_idx, (cand, cand_vec) in enumerate(zip(deduped, candidate_vecs), start=1):
        sim = compute_cosine_similarity(query_vec, cand_vec)
        scored.append(
            ScoredCandidate(
                candidate=cand,
                cosine_similarity=sim,
                search_rank=rank_idx,
            )
        )
    return scored


def build_work_arbiter_prompt(
    combined_artist: str,
    title: str,
    tier: int,
    candidates: Sequence[ScoredCandidate],
) -> str:
    """Build the LLM user prompt for musical work candidate arbitration in Tiers 1-4."""
    tier_labels = {
        1: "Tier 1 (Master Structured)",
        2: "Tier 2 (Release Vinyl Structured)",
        3: "Tier 3 (Master Keyword)",
        4: "Tier 4 (Release Vinyl Keyword)",
    }
    tier_label = tier_labels.get(tier, f"Tier {tier}")

    lines = [
        f"Target Musical Work:\n- Combined Artist: {combined_artist}\n- Album Title: {title}\n",
        f"Search Tier: {tier_label}\n",
        "Discogs Candidates:",
    ]
    if not candidates:
        lines.append("No candidates found in this tier.")
    else:
        for c in candidates:
            format_str = ", ".join(c.format) if c.format else "None"
            lines.append(
                f"- Search Rank: {c.search_rank}\n"
                f"  ID: {c.id}\n"
                f"  Type: {c.type}\n"
                f"  Title: {c.title}\n"
                f"  Master ID: {c.master_id if c.master_id is not None else 'None'}\n"
                f"  Year: {c.year if c.year is not None else 'Unknown'}\n"
                f"  Country: {c.country if c.country is not None else 'Unknown'}\n"
                f"  Format: {format_str}\n"
                f"  Cosine Similarity: {c.cosine_similarity:.4f}"
            )
    return "\n".join(lines)


def build_artist_arbiter_prompt(
    artist_name: str,
    candidates: Sequence[ScoredCandidate],
) -> str:
    """Build the LLM user prompt for constituent artist candidate arbitration in Tier 5."""
    lines = [
        f"Target Artist:\n- Artist Name: {artist_name}\n",
        "Search Tier: Tier 5 (Constituent Artist Fallback)\n",
        "Discogs Candidates:",
    ]
    if not candidates:
        lines.append("No candidates found in this tier.")
    else:
        for c in candidates:
            lines.append(
                f"- Search Rank: {c.search_rank}\n"
                f"  ID: {c.id}\n"
                f"  Type: {c.type}\n"
                f"  Title: {c.title}\n"
                f"  Cosine Similarity: {c.cosine_similarity:.4f}"
            )
    return "\n".join(lines)


def create_candidate_arbiter_agent(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    thinking_level: str | None = None,
    model: Any = None,
) -> Agent[None, CandidateArbiterResult]:
    """Create a PydanticAI Agent configured for candidate arbitration."""
    if model is not None:
        return Agent(
            model,
            output_type=CandidateArbiterResult,
            system_prompt=CANDIDATE_ARBITER_SYSTEM_PROMPT,
        )

    key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
    if not key:
        raise ValueError(
            "DEEPSEEK_API_KEY or DEEPSEEK_KEY environment variable is required"
        )
    url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model_str = model_name or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"

    level = thinking_level or os.getenv("LLM_THINKING_LEVEL")
    if not level:
        level = get_default_thinking_level(model_str)

    model_settings: ModelSettings
    if level.lower() == "none":
        model_settings = {"thinking": False}
    else:
        model_settings = {"thinking": cast(Any, level.lower())}

    provider = OpenAIProvider(base_url=url, api_key=key)
    chat_model = OpenAIChatModel(model_str, provider=provider)

    return Agent(
        chat_model,
        output_type=CandidateArbiterResult,
        system_prompt=CANDIDATE_ARBITER_SYSTEM_PROMPT,
        model_settings=model_settings,
    )


def _run_arbiter_sync(
    agent: Agent[None, CandidateArbiterResult], prompt: str
) -> CandidateArbiterResult:
    """Run agent synchronously, handling active or non-active asyncio event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, agent.run(prompt)).result().output
    else:
        return agent.run_sync(prompt).output


def arbitrate_work_candidates(
    combined_artist: str,
    title: str,
    tier: int,
    candidates: Sequence[ScoredCandidate],
    arbiter_agent: Agent[None, CandidateArbiterResult] | None = None,
) -> ResolvedWorkMatch | None:
    """Evaluate search candidates for a musical work in Tiers 1-4 and apply master_id promotion rules."""
    if not candidates:
        return None

    agent = arbiter_agent or create_candidate_arbiter_agent()
    prompt = build_work_arbiter_prompt(
        combined_artist=combined_artist,
        title=title,
        tier=tier,
        candidates=candidates,
    )
    result = _run_arbiter_sync(agent, prompt)
    if not result.is_match or result.matched_candidate_id is None:
        return None

    winning_scored = next(
        (c for c in candidates if c.id == result.matched_candidate_id),
        None,
    )
    if winning_scored is None:
        logger.warning(
            "Arbiter returned matched_candidate_id=%s not present in candidates list",
            result.matched_candidate_id,
        )
        return None

    cand = winning_scored.candidate

    if cand.type == "master":
        resolved_master_id = cand.id
        resolved_release_id = None
    elif cand.type == "release":
        if cand.master_id:
            resolved_master_id = cand.master_id
            resolved_release_id = cand.id
        else:
            resolved_master_id = None
            resolved_release_id = cand.id
    else:
        resolved_master_id = cand.master_id if cand.master_id else cand.id
        resolved_release_id = cand.id

    return ResolvedWorkMatch(
        tier=tier,
        candidate=cand,
        resolved_master_id=resolved_master_id,
        resolved_release_id=resolved_release_id,
        cosine_similarity=winning_scored.cosine_similarity,
        search_rank=winning_scored.search_rank,
        reasoning=result.reasoning,
    )


def arbitrate_artist_candidates(
    artist_name: str,
    candidates: Sequence[ScoredCandidate],
    arbiter_agent: Agent[None, CandidateArbiterResult] | None = None,
) -> ResolvedArtistMatch | None:
    """Evaluate search candidates for a constituent artist in Tier 5."""
    if not candidates:
        return None

    agent = arbiter_agent or create_candidate_arbiter_agent()
    prompt = build_artist_arbiter_prompt(
        artist_name=artist_name,
        candidates=candidates,
    )
    result = _run_arbiter_sync(agent, prompt)
    if not result.is_match or result.matched_candidate_id is None:
        return None

    winning_scored = next(
        (c for c in candidates if c.id == result.matched_candidate_id),
        None,
    )
    if winning_scored is None:
        logger.warning(
            "Arbiter returned matched artist ID=%s not present in candidates list",
            result.matched_candidate_id,
        )
        return None

    return ResolvedArtistMatch(
        artist_name=artist_name,
        candidate=winning_scored.candidate,
        discogs_artist_id=winning_scored.candidate.id,
        cosine_similarity=winning_scored.cosine_similarity,
        search_rank=winning_scored.search_rank,
        reasoning=result.reasoning,
    )


def _normalize_constituent_artists(
    artists: Sequence[ConstituentArtistInfo | Any | str] | None,
) -> list[ConstituentArtistInfo]:
    """Normalize input constituent artists into ConstituentArtistInfo instances."""
    if not artists:
        return []
    result: list[ConstituentArtistInfo] = []
    for art in artists:
        if isinstance(art, ConstituentArtistInfo):
            result.append(art)
        elif isinstance(art, str):
            result.append(ConstituentArtistInfo(name=art.strip(), discogs_artist_id=None))
        else:
            name = getattr(art, "name", "")
            discogs_id = getattr(art, "discogs_artist_id", None)
            result.append(
                ConstituentArtistInfo(name=str(name).strip(), discogs_artist_id=discogs_id)
            )
    return result


def run_discogs_search_cascade(
    discogs_client: DiscogsClient,
    combined_artist: str = "",
    title: str = "",
    constituent_artists: Sequence[ConstituentArtistInfo | Any | str] | None = None,
    arbiter_agent: Agent[None, CandidateArbiterResult] | None = None,
    album_text: str | None = None,
    album: Any = None,
) -> CascadeResult:
    """Execute up to 5 Discogs search tiers with candidate similarity scoring and LLM arbitration.

    Tiers 1-4 execute in sequence and short-circuit immediately as soon as a tier yields
    a confirmed musical work match. If Tiers 1-4 yield no match, Tier 5 executes per-constituent
    artist searches for any constituent artists lacking an existing discogs_artist_id.
    """
    if album is not None:
        if not combined_artist and hasattr(album, "combined_artist"):
            combined_artist = album.combined_artist
        if not title and hasattr(album, "title"):
            title = album.title
        if constituent_artists is None and hasattr(album, "artists"):
            constituent_artists = album.artists

    agent = arbiter_agent or create_candidate_arbiter_agent()
    work_query_text = album_text or compute_album_text(combined_artist, title)

    tiers_executed: list[int] = []

    # Tier 1 (Master Structured): type="master", artist=combined_artist, release_title=title, per_page=5
    tiers_executed.append(1)
    candidates_t1 = discogs_client.search(
        type="master",
        artist=combined_artist,
        release_title=title,
        per_page=5,
    )
    if candidates_t1:
        scored_t1 = score_candidates(work_query_text, candidates_t1, max_candidates=5)
        match_t1 = arbitrate_work_candidates(
            combined_artist=combined_artist,
            title=title,
            tier=1,
            candidates=scored_t1,
            arbiter_agent=agent,
        )
        if match_t1 is not None:
            return CascadeResult(
                work_match=match_t1,
                tier_reached=1,
                tiers_executed=tiers_executed,
            )

    # Tier 2 (Release Vinyl Structured): type="release", artist=combined_artist, release_title=title, format="Vinyl", per_page=5
    tiers_executed.append(2)
    candidates_t2 = discogs_client.search(
        type="release",
        artist=combined_artist,
        release_title=title,
        format="Vinyl",
        per_page=5,
    )
    if candidates_t2:
        scored_t2 = score_candidates(work_query_text, candidates_t2, max_candidates=5)
        match_t2 = arbitrate_work_candidates(
            combined_artist=combined_artist,
            title=title,
            tier=2,
            candidates=scored_t2,
            arbiter_agent=agent,
        )
        if match_t2 is not None:
            return CascadeResult(
                work_match=match_t2,
                tier_reached=2,
                tiers_executed=tiers_executed,
            )

    # Tier 3 (Master Keyword): type="master", q=f"{combined_artist} {title}", per_page=5
    tiers_executed.append(3)
    candidates_t3 = discogs_client.search(
        type="master",
        q=f"{combined_artist} {title}",
        per_page=5,
    )
    if candidates_t3:
        scored_t3 = score_candidates(work_query_text, candidates_t3, max_candidates=5)
        match_t3 = arbitrate_work_candidates(
            combined_artist=combined_artist,
            title=title,
            tier=3,
            candidates=scored_t3,
            arbiter_agent=agent,
        )
        if match_t3 is not None:
            return CascadeResult(
                work_match=match_t3,
                tier_reached=3,
                tiers_executed=tiers_executed,
            )

    # Tier 4 (Release Vinyl Keyword): type="release", q=f"{combined_artist} {title}", format="Vinyl", per_page=5
    tiers_executed.append(4)
    candidates_t4 = discogs_client.search(
        type="release",
        q=f"{combined_artist} {title}",
        format="Vinyl",
        per_page=5,
    )
    if candidates_t4:
        scored_t4 = score_candidates(work_query_text, candidates_t4, max_candidates=5)
        match_t4 = arbitrate_work_candidates(
            combined_artist=combined_artist,
            title=title,
            tier=4,
            candidates=scored_t4,
            arbiter_agent=agent,
        )
        if match_t4 is not None:
            return CascadeResult(
                work_match=match_t4,
                tier_reached=4,
                tiers_executed=tiers_executed,
            )

    # Tier 5 (Per-Constituent Artist Fallback): type="artist", q=artist_name, per_page=5
    tiers_executed.append(5)
    normalized_artists = _normalize_constituent_artists(constituent_artists)
    artist_matches: dict[str, ResolvedArtistMatch] = {}

    for artist_info in normalized_artists:
        if artist_info.discogs_artist_id is not None and artist_info.discogs_artist_id > 0:
            logger.debug(
                "Skipping Tier 5 search for artist '%s' (already linked with discogs_artist_id=%d)",
                artist_info.name,
                artist_info.discogs_artist_id,
            )
            continue

        if not artist_info.name:
            continue

        candidates_art = discogs_client.search(
            type="artist",
            q=artist_info.name,
            per_page=5,
        )
        if candidates_art:
            art_query_text = compute_artist_text(artist_info.name)
            scored_art = score_candidates(art_query_text, candidates_art, max_candidates=5)
            art_match = arbitrate_artist_candidates(
                artist_name=artist_info.name,
                candidates=scored_art,
                arbiter_agent=agent,
            )
            if art_match is not None:
                artist_matches[artist_info.name] = art_match

    return CascadeResult(
        work_match=None,
        artist_matches=artist_matches,
        tier_reached=5,
        tiers_executed=tiers_executed,
    )


__all__ = [
    "CANDIDATE_ARBITER_SYSTEM_PROMPT",
    "CandidateArbiterResult",
    "CascadeResult",
    "ConstituentArtistInfo",
    "ResolvedArtistMatch",
    "ResolvedWorkMatch",
    "ScoredCandidate",
    "arbitrate_artist_candidates",
    "arbitrate_work_candidates",
    "build_artist_arbiter_prompt",
    "build_work_arbiter_prompt",
    "compute_cosine_similarity",
    "create_candidate_arbiter_agent",
    "deduplicate_candidates",
    "run_discogs_search_cascade",
    "score_candidates",
]




