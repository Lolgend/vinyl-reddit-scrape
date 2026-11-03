"""Discogs catalog enrichment, multi-tier search cascade, and LLM candidate arbitration."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import os
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar, cast

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from sqlalchemy import and_, case, delete, func, literal, or_, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from reddit_scrape.discogs import (
    DiscogsAPIError,
    DiscogsArtistCredit,
    DiscogsClient,
    DiscogsSearchCandidate,
)
from reddit_scrape.embeddings import (
    compute_album_text,
    compute_artist_text,
    embed_texts,
)
from reddit_scrape.extraction import get_default_thinking_level
from reddit_scrape.models import (
    Album,
    AlbumArtist,
    Artist,
    EnrichmentState,
    Release,
    SubmissionRelease,
)
from reddit_scrape.persistence import (
    compute_album_fingerprint,
    compute_release_fingerprint,
    normalize_attribute,
)

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

HOMONYM_DISAMBIGUATION_SYSTEM_PROMPT = """You are an expert discography classifier and catalog reconciliation agent.
Your task is to take hydrated Discogs work metadata (work title, artists[] credits including id, name, anv, join) and produce clean canonical catalog metadata.

Rules:
1. Canonical Artist Homonym Disambiguation:
   - Discogs appends numeric homonym suffixes in parentheses to artist names to distinguish different artists with the same name (e.g., 'Nirvana (2)', 'Blink (3)', 'The Smile (5)', 'Death (2)').
   - Strip numeric homonym suffixes like '(2)', '(3)' to produce the clean artist name (e.g., 'Nirvana', 'Blink', 'The Smile', 'Death').
   - PRESERVE legitimate parenthetical or symbolic names that are an integral part of the actual band name (e.g., 'Sunn O)))', 'Someone (Somewhere)'). Never strip non-numeric or legitimate names.
   - Pair each cleaned canonical artist name with its exact `discogs_artist_id`.

2. Combined Artist Billing Reconstruction:
   - Reconstruct the official combined artist billing using the `join` delimiters provided in the Discogs artists array.
   - For example:
     - Artist 1 'Converge (2)' (join='/') + Artist 2 'Dropdead' -> 'Converge / Dropdead'
     - Artist 1 'Kanye West' (join='&') + Artist 2 'Ty Dolla $ign' -> 'Kanye West & Ty Dolla $ign'
     - Artist 1 'Artist A' (join='feat.') + Artist 2 'Artist B' -> 'Artist A feat. Artist B'
   - If join is empty or missing between multiple artists, default to ' / '.
   - If there is only one artist, combined_artist is just the cleaned artist name.

3. Canonical Album Title:
   - Return the official canonical work title from Discogs, cleaning any extraneous whitespace or obvious formatting errors while preserving official punctuation.
"""


WEB_RECONCILIATION_SYSTEM_PROMPT = """You are an expert discography classifier and catalog reconciliation agent.
Your task is to evaluate native web search results for a music release (album, EP, single) not yet indexed on Discogs and produce clean, standardized canonical catalog metadata.

Rules:
1. Canonical Combined Artist & Constituent Artists:
   - Identify the primary performing artist(s) from the search results.
   - Separate individual constituent artists in `canonical_artists` (e.g. ['Converge', 'Chelsea Wolfe']).
   - Produce the standardized `canonical_combined_artist` billing (e.g. 'Converge & Chelsea Wolfe', 'Dropdead / Converge', or single artist 'Kendrick Lamar').
   - Standardize official capitalization, diacritics, and official artist spelling.

2. Canonical Album Title:
   - Extract the official canonical work title (e.g. 'Jane Doe', 'GNX', 'Untitled 2027 Studio LP').
   - Strictly strip retailer, colorway, vinyl pressing weight, run numbering, and packaging descriptors (e.g. '(Target Exclusive Opaque Gold)', '(2LP Gatefold 180g Black)').
   - Preserve integral subtitles or volume numbers (e.g. 'Bloodmoon: I').

3. Fallback Integrity:
   - If the search results are ambiguous or tangential, favor the extracted artist and title names while cleaning obvious promotional noise.
"""


SIBLING_RECONCILIATION_SYSTEM_PROMPT = """You are an expert discography classifier and catalog reconciliation agent.
Your task is to evaluate sibling physical Release pressings belonging to the same canonical Album and identify duplicate entries that represent the exact same commercial pressing/variant expressed with different wording.

Evaluation Rules:
1. Identify Duplicate Pressings:
   - Group rows that refer to the exact same physical release variant and retailer (e.g., 'Pink /500' vs 'Limited Pink Vinyl (500 copies)', 'Standard Black' vs 'Black Vinyl', or 'Target Exclusive Opaque Red' vs 'Target Red').
   - Do NOT merge genuinely distinct pressings (e.g., 'Pink Vinyl /500' vs 'Clear Vinyl /100', '1LP Standard' vs '2LP Deluxe 45RPM', or different retailers like 'Rough Trade' vs 'Target').
   - If in doubt, keep them separate. Only group items that are clearly identical commercial releases described with slightly different phrasing.

2. Canonical Standardized Output:
   - For each group of duplicate releases:
     - Set `canonical_release_id` to the ID of the release to keep.
     - Set `duplicate_release_ids` to the list of duplicate release IDs to merge into `canonical_release_id`.
     - Set `standardized_variant` to the clean standardized variant description (e.g., 'Pink /500').
     - Set `standardized_retailer` to the clean standardized retailer name (e.g., 'Deathwish Inc').
   - If all siblings are distinct pressings, return `duplicate_groups = []`.
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


class CanonicalArtistCredit(BaseModel):
    """Canonical constituent artist name and Discogs artist ID."""

    discogs_artist_id: int = Field(description="The Discogs artist ID.")
    name: str = Field(
        description="Clean canonical artist name, with Discogs numeric homonym suffix (e.g. '(2)') stripped, but legitimate parenthetical names preserved."
    )


class WorkReconciliationResult(BaseModel):
    """Structured LLM reconciliation of Discogs work title and artist credits."""

    canonical_combined_artist: str = Field(
        description="Reconstructed combined artist billing using Discogs artists[].join delimiters (e.g. 'Converge / Dropdead', 'Kanye West & Ty Dolla $ign')."
    )
    canonical_artists: list[CanonicalArtistCredit] = Field(
        description="List of canonical artist credits with clean names and discogs_artist_id."
    )
    canonical_album_title: str = Field(
        description="Clean canonical album title from the Discogs work title."
    )
    reasoning: str | None = Field(
        None,
        description="Brief explanation of homonym stripping and billing reconstruction decisions.",
    )


class WebReconciliationResult(BaseModel):
    """Structured LLM reconciliation of web search results for unindexed releases."""

    canonical_combined_artist: str = Field(
        description="Standardized canonical combined artist billing."
    )
    canonical_artists: list[str] = Field(
        description="Standardized canonical individual constituent artist names."
    )
    canonical_album_title: str = Field(
        description="Standardized canonical musical work / album title."
    )
    reasoning: str | None = Field(
        default=None,
        description="Brief explanation of canonicalization decisions.",
    )


class SiblingDuplicateGroup(BaseModel):
    """A group of duplicate sibling release rows to merge into a single canonical release."""

    canonical_release_id: str = Field(
        description="The ID of the surviving canonical release row."
    )
    duplicate_release_ids: list[str] = Field(
        description="The IDs of the duplicate release rows to merge into the canonical release."
    )
    standardized_variant: str | None = Field(
        default=None,
        description="Clean standardized variant description (e.g. 'Pink /500').",
    )
    standardized_retailer: str | None = Field(
        default=None,
        description="Clean standardized retailer name (e.g. 'Deathwish Inc').",
    )


class SiblingReconciliationResult(BaseModel):
    """Structured LLM result identifying duplicate sibling release groups."""

    duplicate_groups: list[SiblingDuplicateGroup] = Field(
        default_factory=list,
        description="List of duplicate groups to merge. Empty if all sibling releases are distinct.",
    )
    reasoning: str | None = Field(
        default=None,
        description="Brief explanation of sibling release grouping decisions.",
    )


@dataclass
class SiblingReleaseInfo:
    """Sibling release metadata passed to Phase 4 LLM reconciliation."""

    id: uuid.UUID | str
    variant: str | None = None
    retailer_name: str | None = None


@dataclass
class WebSearchHit:
    """A web search hit returned by DeepSeek native web search tool (POST /v1/tools/search)."""

    title: str
    url: str
    summary: str = ""
    snippet: str = ""
    content: str = ""
    rerank_score: float | None = None


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
class HydratedWorkDetail:
    """Detail payload retrieved from a follow-up GET /masters/{id} or GET /releases/{id} call."""

    resolved_master_id: int | None
    resolved_release_id: int | None
    title: str
    artists: list[DiscogsArtistCredit]
    year: int | None = None
    genres: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)


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
class PendingConstituentArtist:
    """Constituent artist retrieved during Phase 1 read."""

    id: uuid.UUID
    name: str
    discogs_artist_id: int | None = None


@dataclass
class PendingAlbumData:
    """Pending album metadata and linked artists retrieved during Phase 1 read."""

    id: uuid.UUID
    combined_artist: str
    title: str
    fingerprint: str
    discogs_status: str
    artists: list[PendingConstituentArtist] = field(default_factory=list)


@dataclass
class EnrichedArtistData:
    """Canonical constituent artist prepared for Phase 3 write."""

    artist_id: uuid.UUID | None
    name: str
    embedding: list[float]
    discogs_artist_id: int | None = None


@dataclass
class EnrichedAlbumData:
    """Canonical album and artists prepared during Phase 2 for Phase 3 commit."""

    album_id: uuid.UUID
    combined_artist: str
    title: str
    fingerprint: str
    discogs_master_id: int | None
    discogs_release_id: int | None
    discogs_status: str
    embedding: list[float]
    artists: list[EnrichedArtistData] = field(default_factory=list)
    reasoning: str | None = None


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


def search_web_for_release(
    query: str,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.Client | None = None,
) -> list[WebSearchHit]:
    """Execute a native web search call to POST {deepseek_base_url}/v1/tools/search.

    Sends:
        {"query": query[:128], "count": 10, "need_content": True}
    Returns search hits ordered by rerank_score descending.
    Gracefully returns an empty list on HTTP errors or network failures.
    """
    key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
    if not key:
        logger.warning("No DeepSeek API key found; skipping native web search")
        return []

    raw_url = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    endpoint = f"{raw_url.rstrip('/')}/v1/tools/search"

    truncated_query = query[:128]
    payload: dict[str, Any] = {
        "query": truncated_query,
        "count": 10,
        "need_content": True,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    client = http_client or httpx.Client(timeout=30.0)
    close_client = http_client is None

    try:
        response = client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("DeepSeek web search call failed for query '%s': %s", truncated_query, exc)
        return []
    finally:
        if close_client:
            client.close()

    raw_hits: list[dict[str, Any]] = []
    if isinstance(data, list):
        raw_hits = data
    elif isinstance(data, dict):
        for key_name in ("results", "data", "hits"):
            candidate_list = data.get(key_name)
            if isinstance(candidate_list, list):
                raw_hits = candidate_list
                break

    hits: list[WebSearchHit] = []
    for raw in raw_hits:
        if not isinstance(raw, dict):
            continue
        score_val = raw.get("rerank_score") or raw.get("score")
        rerank_score = float(score_val) if score_val is not None else None
        hits.append(
            WebSearchHit(
                title=str(raw.get("title") or raw.get("name") or "").strip(),
                url=str(raw.get("url") or raw.get("link") or "").strip(),
                summary=str(raw.get("summary") or "").strip(),
                snippet=str(raw.get("snippet") or "").strip(),
                content=str(raw.get("content") or raw.get("text") or "").strip(),
                rerank_score=rerank_score,
            )
        )

    # Order by rerank_score descending (None treated as lowest)
    hits.sort(key=lambda h: (h.rerank_score is not None, h.rerank_score or -float("inf")), reverse=True)
    return hits


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


_AgentResultT = TypeVar("_AgentResultT")


def _create_deepseek_agent(
    output_type: type[_AgentResultT],
    system_prompt: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    thinking_level: str | None = None,
    model: Any = None,
) -> Agent[None, _AgentResultT]:
    """Shared factory for PydanticAI agents configured with DeepSeek chat model/provider/settings."""
    if model is not None:
        return Agent(
            model,
            output_type=output_type,
            system_prompt=system_prompt,
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
        output_type=output_type,
        system_prompt=system_prompt,
        model_settings=model_settings,
    )


def create_candidate_arbiter_agent(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    thinking_level: str | None = None,
    model: Any = None,
) -> Agent[None, CandidateArbiterResult]:
    """Create a PydanticAI Agent configured for candidate arbitration."""
    return _create_deepseek_agent(
        output_type=CandidateArbiterResult,
        system_prompt=CANDIDATE_ARBITER_SYSTEM_PROMPT,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        thinking_level=thinking_level,
        model=model,
    )


def _run_agent_sync(
    agent: Agent[Any, Any], prompt: str
) -> Any:
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


_run_arbiter_sync = _run_agent_sync


def reconstruct_combined_artist(
    artists: Sequence[DiscogsArtistCredit],
    canonical_names: Sequence[str],
) -> str:
    """Reconstruct canonical combined artist billing using Discogs artists[].join delimiters."""
    parts: list[str] = []
    for idx, art in enumerate(artists):
        name = canonical_names[idx] if idx < len(canonical_names) else art.name
        join_tok = (art.join or "").strip()
        parts.append(name)
        if idx < len(artists) - 1:
            if join_tok:
                delim = f"{join_tok} " if join_tok in (",", ";") else f" {join_tok} "
                parts.append(delim)
            else:
                parts.append(" / ")
    return "".join(parts).strip()


def build_homonym_disambiguation_prompt(detail: HydratedWorkDetail) -> str:
    """Build the LLM user prompt for artist homonym disambiguation and billing reconstruction."""
    lines = [
        "Hydrated Discogs Work Detail:",
        f"- Work Title: {detail.title}",
        f"- Master ID: {detail.resolved_master_id if detail.resolved_master_id is not None else 'None'}",
        f"- Release ID: {detail.resolved_release_id if detail.resolved_release_id is not None else 'None'}",
        f"- Year: {detail.year if detail.year is not None else 'Unknown'}",
        "",
        "Contributing Artists Credit Array:",
    ]
    if not detail.artists:
        lines.append("No artist credits provided in payload.")
    else:
        for idx, art in enumerate(detail.artists, start=1):
            join_str = f"'{art.join}'" if art.join else "None"
            anv_str = f"'{art.anv}'" if art.anv else "None"
            lines.append(
                f"{idx}. ID: {art.id}\n"
                f"   Name: {art.name}\n"
                f"   ANV: {anv_str}\n"
                f"   Join: {join_str}"
            )
    return "\n".join(lines)


def create_homonym_disambiguation_agent(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    thinking_level: str | None = None,
    model: Any = None,
) -> Agent[None, WorkReconciliationResult]:
    """Create a PydanticAI Agent configured for artist homonym disambiguation and billing reconstruction."""
    return _create_deepseek_agent(
        output_type=WorkReconciliationResult,
        system_prompt=HOMONYM_DISAMBIGUATION_SYSTEM_PROMPT,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        thinking_level=thinking_level,
        model=model,
    )


def reconcile_artist_homonyms(
    detail: HydratedWorkDetail,
    disambiguation_agent: Agent[None, WorkReconciliationResult] | None = None,
) -> WorkReconciliationResult:
    """Pass hydrated Discogs work metadata through LLM to strip numeric homonyms and reconstruct billing."""
    agent = disambiguation_agent or create_homonym_disambiguation_agent()
    prompt = build_homonym_disambiguation_prompt(detail)
    return cast(WorkReconciliationResult, _run_agent_sync(agent, prompt))


def build_web_reconciliation_prompt(
    combined_artist: str,
    album_title: str,
    hits: Sequence[WebSearchHit],
    max_content_chars_per_hit: int = 1000,
) -> str:
    """Build LLM user prompt for web search reconciliation."""
    lines = [
        "Target Extracted Release:",
        f"- Combined Artist: {combined_artist}",
        f"- Album Title: {album_title}",
        "",
        "Web Search Hits (ordered by rerank score):",
    ]
    if not hits:
        lines.append("No web search hits returned.")
    else:
        for idx, hit in enumerate(hits, start=1):
            score_str = f"{hit.rerank_score:.4f}" if hit.rerank_score is not None else "N/A"
            content_excerpt = hit.content[:max_content_chars_per_hit].strip() if hit.content else ""
            lines.append(
                f"{idx}. Title: {hit.title}\n"
                f"   URL: {hit.url}\n"
                f"   Rerank Score: {score_str}\n"
                f"   Summary: {hit.summary}\n"
                f"   Snippet: {hit.snippet}\n"
                f"   Content Excerpt: {content_excerpt}"
            )
    return "\n".join(lines)


def create_web_reconciliation_agent(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    thinking_level: str | None = None,
    model: Any = None,
) -> Agent[None, WebReconciliationResult]:
    """Create a PydanticAI Agent configured for web search reconciliation."""
    return _create_deepseek_agent(
        output_type=WebReconciliationResult,
        system_prompt=WEB_RECONCILIATION_SYSTEM_PROMPT,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        thinking_level=thinking_level,
        model=model,
    )


def reconcile_web_search(
    combined_artist: str,
    album_title: str,
    hits: Sequence[WebSearchHit],
    web_agent: Agent[None, WebReconciliationResult] | None = None,
    raw_artists: Sequence[str] | None = None,
) -> WebReconciliationResult:
    """Pass web search results through LLM to standardize canonical artist/album names.

    Gracefully falls back to extracted names if hits is empty or agent execution errors.
    """
    fallback_artists = (
        [a.strip() for a in raw_artists if a.strip()]
        if raw_artists
        else [combined_artist]
    )
    if not fallback_artists:
        fallback_artists = [combined_artist]

    fallback = WebReconciliationResult(
        canonical_combined_artist=combined_artist,
        canonical_artists=fallback_artists,
        canonical_album_title=album_title,
        reasoning="Fallback to Pass 1 extracted names (0 search hits or error)",
    )
    if not hits:
        return fallback

    agent = web_agent or create_web_reconciliation_agent()
    prompt = build_web_reconciliation_prompt(
        combined_artist=combined_artist,
        album_title=album_title,
        hits=hits,
    )
    try:
        return cast(WebReconciliationResult, _run_agent_sync(agent, prompt))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Web search reconciliation agent failed for '%s - %s': %s. Retaining extracted names.",
            combined_artist,
            album_title,
            exc,
        )
        return fallback


def build_sibling_reconciliation_prompt(
    combined_artist: str,
    album_title: str,
    siblings: Sequence[SiblingReleaseInfo | Any],
) -> str:
    """Build the LLM user prompt for sibling Release reconciliation."""
    lines = [
        f"Parent Canonical Album:\n- Combined Artist: {combined_artist}\n- Album Title: {album_title}\n",
        "Sibling Physical Releases:",
    ]
    for s in siblings:
        s_id = str(getattr(s, "id", ""))
        variant = getattr(s, "variant", None) or "None"
        retailer = getattr(s, "retailer_name", None) or "None"
        lines.append(
            f"- ID: {s_id}\n"
            f"  Variant: {variant}\n"
            f"  Retailer: {retailer}"
        )
    return "\n".join(lines)


def create_sibling_reconciliation_agent(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    thinking_level: str | None = None,
    model: Any = None,
) -> Agent[None, SiblingReconciliationResult]:
    """Create a PydanticAI Agent configured for sibling release reconciliation."""
    return _create_deepseek_agent(
        output_type=SiblingReconciliationResult,
        system_prompt=SIBLING_RECONCILIATION_SYSTEM_PROMPT,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        thinking_level=thinking_level,
        model=model,
    )


def reconcile_sibling_releases(
    combined_artist: str,
    album_title: str,
    siblings: Sequence[SiblingReleaseInfo | Any],
    sibling_agent: Agent[None, SiblingReconciliationResult] | None = None,
) -> SiblingReconciliationResult:
    """Pass sibling releases to LLM outside any DB session to detect duplicate (variant, retailer) pressings."""
    if len(siblings) < 2:
        return SiblingReconciliationResult(duplicate_groups=[])

    agent = sibling_agent or create_sibling_reconciliation_agent()
    prompt = build_sibling_reconciliation_prompt(
        combined_artist=combined_artist,
        album_title=album_title,
        siblings=siblings,
    )
    try:
        return cast(SiblingReconciliationResult, _run_agent_sync(agent, prompt))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sibling release reconciliation agent failed for '%s - %s': %s",
            combined_artist,
            album_title,
            exc,
        )
        return SiblingReconciliationResult(duplicate_groups=[], reasoning=str(exc))


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


def hydrate_work_detail(
    discogs_client: DiscogsClient,
    work_match: ResolvedWorkMatch,
) -> HydratedWorkDetail:
    """Execute 1 follow-up detail call to retrieve structured artists[] and canonical work title.

    Calls GET /masters/{resolved_master_id} when resolved_master_id is present, recording
    main_release as resolved_release_id when available and not already set.
    Calls GET /releases/{resolved_release_id} when the release is unmastered.
    """
    if work_match.resolved_master_id is not None:
        try:
            master = discogs_client.get_master(work_match.resolved_master_id)
            resolved_rel_id = work_match.resolved_release_id
            if resolved_rel_id is None and master.main_release is not None:
                resolved_rel_id = master.main_release
            return HydratedWorkDetail(
                resolved_master_id=work_match.resolved_master_id,
                resolved_release_id=resolved_rel_id,
                title=master.title,
                artists=master.artists,
                year=master.year,
                genres=master.genres,
                styles=master.styles,
            )
        except DiscogsAPIError:
            if work_match.resolved_release_id is None:
                raise
            release = discogs_client.get_release(work_match.resolved_release_id)
            return HydratedWorkDetail(
                resolved_master_id=None,
                resolved_release_id=work_match.resolved_release_id,
                title=release.title,
                artists=release.artists,
                year=release.year,
                genres=release.genres,
                styles=release.styles,
            )

    if work_match.resolved_release_id is not None:
        release = discogs_client.get_release(work_match.resolved_release_id)
        return HydratedWorkDetail(
            resolved_master_id=release.master_id,
            resolved_release_id=work_match.resolved_release_id,
            title=release.title,
            artists=release.artists,
            year=release.year,
            genres=release.genres,
            styles=release.styles,
        )

    raise ValueError("Cannot hydrate work detail without resolved_master_id or resolved_release_id")


def _clean_homonym_suffix(name: str) -> str:
    """Strip numeric homonym suffix (e.g. 'Artist (2)' -> 'Artist')."""
    return re.sub(r"\s+\(\d+\)$", "", name).strip()


def get_next_queued_album(
    session: Session,
    now: datetime | None = None,
) -> Album | None:
    """Query the next eligible Album from partial index idx_albums_enrichment_queue.

    Priority levels:
    - P0 ('pending'): discogs_status = 'pending'
    - P1 ('failed'): discogs_status = 'failed' AND enrichment_attempts < 5
                    AND (last_enriched_at IS NULL OR last_enriched_at <= now - (5m * 2^(enrichment_attempts - 1)))
    - P2 ('partial_matched'): discogs_status = 'partial_matched'
                             AND (last_enriched_at IS NULL OR last_enriched_at <= now - 7 days)
                             AND created_at >= now - 60 days

    Ordering:
    CASE discogs_status WHEN 'pending' THEN 0 WHEN 'failed' THEN 1 WHEN 'partial_matched' THEN 2 ELSE 9 END ASC,
    created_at ASC,
    last_enriched_at ASC NULLS FIRST
    """
    now_expr = literal(now) if now is not None else func.now()

    pending_condition = Album.discogs_status == EnrichmentState.PENDING.value

    due_failed_condition = and_(
        Album.discogs_status == EnrichmentState.FAILED.value,
        Album.enrichment_attempts < 5,
        or_(
            Album.last_enriched_at.is_(None),
            Album.last_enriched_at
            <= now_expr
            - (
                text("interval '5 minutes'")
                * func.power(2, Album.enrichment_attempts - 1)
            ),
        ),
    )

    due_partial_matched_condition = and_(
        Album.discogs_status == EnrichmentState.PARTIAL_MATCHED.value,
        or_(
            Album.last_enriched_at.is_(None),
            Album.last_enriched_at <= now_expr - text("interval '7 days'"),
        ),
        Album.created_at >= now_expr - text("interval '60 days'"),
    )

    priority_case = case(
        (Album.discogs_status == EnrichmentState.PENDING.value, 0),
        (Album.discogs_status == EnrichmentState.FAILED.value, 1),
        (Album.discogs_status == EnrichmentState.PARTIAL_MATCHED.value, 2),
        else_=9,
    )

    stmt = (
        select(Album)
        .where(
            or_(
                pending_condition,
                due_failed_condition,
                due_partial_matched_condition,
            )
        )
        .order_by(
            priority_case.asc(),
            Album.created_at.asc(),
            Album.last_enriched_at.asc().nulls_first(),
        )
        .limit(1)
    )

    return session.execute(stmt).scalars().first()


def _resolve_constituent_artists(
    pending_artists: Sequence[PendingConstituentArtist],
    canonical_artists: Sequence[CanonicalArtistCredit | str],
    artist_embeddings: Sequence[list[float]],
    cascade_artist_matches: dict[str, ResolvedArtistMatch] | None = None,
    preserve_remaining_pending: bool = False,
) -> list[EnrichedArtistData]:
    """Resolve pending constituent artists against canonical artist credits or names."""
    enriched_artists: list[EnrichedArtistData] = []
    remaining_pending = list(pending_artists)
    total_canonical = len(canonical_artists)

    for idx, (canonical_artist, vec) in enumerate(
        zip(canonical_artists, artist_embeddings)
    ):
        if isinstance(canonical_artist, CanonicalArtistCredit):
            canonical_name = canonical_artist.name
            discogs_id: int | None = canonical_artist.discogs_artist_id
        else:
            canonical_name = str(canonical_artist)
            discogs_id = None

        norm_canonical = normalize_attribute(canonical_name)
        norm_canonical_clean = normalize_attribute(_clean_homonym_suffix(canonical_name))

        # 1. Exact match on normalized name
        matched_pending = next(
            (
                pending_artist
                for pending_artist in remaining_pending
                if normalize_attribute(pending_artist.name) == norm_canonical
            ),
            None,
        )
        # 2. Match after stripping homonym suffixes (n)
        if matched_pending is None:
            matched_pending = next(
                (
                    pending_artist
                    for pending_artist in remaining_pending
                    if normalize_attribute(_clean_homonym_suffix(pending_artist.name))
                    == norm_canonical_clean
                ),
                None,
            )
        # 3. Match by position if remaining counts align
        if (
            matched_pending is None
            and len(remaining_pending) == (total_canonical - idx)
        ) or (
            matched_pending is None
            and len(remaining_pending) == 1
        ):
            matched_pending = remaining_pending[0]

        if matched_pending is not None:
            remaining_pending.remove(matched_pending)
            matched_artist_id = matched_pending.id
            if discogs_id is None:
                discogs_id = matched_pending.discogs_artist_id
            if (
                discogs_id is None
                and cascade_artist_matches
                and matched_pending.name in cascade_artist_matches
            ):
                discogs_id = cascade_artist_matches[
                    matched_pending.name
                ].discogs_artist_id
        else:
            matched_artist_id = None
            if (
                discogs_id is None
                and cascade_artist_matches
                and canonical_name in cascade_artist_matches
            ):
                discogs_id = cascade_artist_matches[
                    canonical_name
                ].discogs_artist_id

        enriched_artists.append(
            EnrichedArtistData(
                artist_id=matched_artist_id,
                name=canonical_name,
                discogs_artist_id=discogs_id,
                embedding=vec,
            )
        )

    if preserve_remaining_pending:
        for pending_artist in remaining_pending:
            pending_vec = embed_texts([compute_artist_text(pending_artist.name)])[0]
            discogs_id = pending_artist.discogs_artist_id
            if (
                discogs_id is None
                and cascade_artist_matches
                and pending_artist.name in cascade_artist_matches
            ):
                discogs_id = cascade_artist_matches[
                    pending_artist.name
                ].discogs_artist_id
            enriched_artists.append(
                EnrichedArtistData(
                    artist_id=pending_artist.id,
                    name=pending_artist.name,
                    discogs_artist_id=discogs_id,
                    embedding=pending_vec,
                )
            )

    return enriched_artists


class CatalogEnrichmentService:
    """Orchestrates multi-phase detached Discogs catalog enrichment."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] | Any,
        discogs_client: DiscogsClient,
        arbiter_agent: Agent[None, CandidateArbiterResult] | None = None,
        disambiguation_agent: Agent[None, WorkReconciliationResult] | None = None,
        web_agent: Agent[None, WebReconciliationResult] | None = None,
        sibling_agent: Agent[None, SiblingReconciliationResult] | None = None,
        deepseek_api_key: str | None = None,
        deepseek_base_url: str | None = None,
        deepseek_http_client: httpx.Client | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.discogs_client = discogs_client
        self.arbiter_agent = arbiter_agent
        self.disambiguation_agent = disambiguation_agent
        self.web_agent = web_agent
        self.sibling_agent = sibling_agent
        self.deepseek_api_key = deepseek_api_key
        self.deepseek_base_url = deepseek_base_url
        self.deepseek_http_client = deepseek_http_client

    def fetch_next_queued_album(self, now: datetime | None = None) -> PendingAlbumData | None:
        """Query the next eligible album from the queue and return its metadata in a short read session."""
        with self.session_factory() as session:
            album = get_next_queued_album(session, now=now)
            if album is None:
                return None
            album_id = album.id
            stmt = (
                select(Artist)
                .join(AlbumArtist, AlbumArtist.artist_id == Artist.id)
                .where(AlbumArtist.album_id == album_id)
                .order_by(AlbumArtist.created_at)
            )
            artists = session.execute(stmt).scalars().all()
            artist_items = [
                PendingConstituentArtist(
                    id=cast(uuid.UUID, a.id),
                    name=str(a.name),
                    discogs_artist_id=cast("int | None", a.discogs_artist_id),
                )
                for a in artists
            ]
            return PendingAlbumData(
                id=cast(uuid.UUID, album.id),
                combined_artist=str(album.combined_artist),
                title=str(album.title),
                fingerprint=str(album.fingerprint),
                discogs_status=str(album.discogs_status),
                artists=artist_items,
            )

    def fetch_pending_album(self, album_id: uuid.UUID | str | Any) -> PendingAlbumData | None:
        """Phase 1: Fetch pending album and its constituent artists in a short read session."""
        uid = uuid.UUID(str(album_id)) if not isinstance(album_id, uuid.UUID) else album_id
        with self.session_factory() as session:
            album = session.get(Album, uid)
            if album is None:
                return None
            stmt = (
                select(Artist)
                .join(AlbumArtist, AlbumArtist.artist_id == Artist.id)
                .where(AlbumArtist.album_id == uid)
                .order_by(AlbumArtist.created_at)
            )
            artists = session.execute(stmt).scalars().all()
            artist_items = [
                PendingConstituentArtist(
                    id=cast(uuid.UUID, a.id),
                    name=str(a.name),
                    discogs_artist_id=cast("int | None", a.discogs_artist_id),
                )
                for a in artists
            ]
            return PendingAlbumData(
                id=cast(uuid.UUID, album.id),
                combined_artist=str(album.combined_artist),
                title=str(album.title),
                fingerprint=str(album.fingerprint),
                discogs_status=str(album.discogs_status),
                artists=artist_items,
            )

    def execute_detached_enrichment(self, pending: PendingAlbumData) -> EnrichedAlbumData | None:
        """Phase 2: Execute Discogs cascade, hydration, LLM reconciliation, and FastEmbed vectors with zero open DB session."""
        constituent_inputs = [
            ConstituentArtistInfo(name=a.name, discogs_artist_id=a.discogs_artist_id)
            for a in pending.artists
        ]
        cascade_result = run_discogs_search_cascade(
            discogs_client=self.discogs_client,
            combined_artist=pending.combined_artist,
            title=pending.title,
            constituent_artists=constituent_inputs,
            arbiter_agent=self.arbiter_agent,
        )

        if cascade_result.work_match is not None:
            detail = hydrate_work_detail(
                discogs_client=self.discogs_client,
                work_match=cascade_result.work_match,
            )

            reconciled = reconcile_artist_homonyms(
                detail=detail,
                disambiguation_agent=self.disambiguation_agent,
            )

            canonical_combined = reconciled.canonical_combined_artist
            canonical_title = reconciled.canonical_album_title
            canonical_artists = reconciled.canonical_artists

            album_text = compute_album_text(canonical_combined, canonical_title)
            artist_texts = [compute_artist_text(ca.name) for ca in canonical_artists]
            all_texts = [album_text] + artist_texts
            vectors = embed_texts(all_texts)
            album_vec = vectors[0]
            artist_vecs = vectors[1:]

            canonical_fp = compute_album_fingerprint(canonical_combined, canonical_title)

            enriched_artists = _resolve_constituent_artists(
                pending_artists=pending.artists,
                canonical_artists=canonical_artists,
                artist_embeddings=artist_vecs,
                cascade_artist_matches=cascade_result.artist_matches,
                preserve_remaining_pending=False,
            )

            return EnrichedAlbumData(
                album_id=pending.id,
                combined_artist=canonical_combined,
                title=canonical_title,
                fingerprint=canonical_fp,
                discogs_master_id=detail.resolved_master_id,
                discogs_release_id=detail.resolved_release_id,
                discogs_status="matched",
                embedding=album_vec,
                artists=enriched_artists,
                reasoning=reconciled.reasoning,
            )

        # Line 3 DeepSeek Native Web Search Fallback (Tiers 1-4 unmatched)
        has_discogs_artist = any(
            a.discogs_artist_id is not None and a.discogs_artist_id > 0
            for a in pending.artists
        ) or any(
            m.discogs_artist_id > 0 for m in cascade_result.artist_matches.values()
        )
        target_status = "partial_matched" if has_discogs_artist else "not_found"

        search_query = f"{pending.combined_artist} {pending.title} vinyl album"
        hits = search_web_for_release(
            query=search_query,
            api_key=self.deepseek_api_key,
            base_url=self.deepseek_base_url,
            http_client=self.deepseek_http_client,
        )

        web_reconciled = reconcile_web_search(
            combined_artist=pending.combined_artist,
            album_title=pending.title,
            hits=hits,
            web_agent=self.web_agent,
            raw_artists=[a.name for a in pending.artists],
        )

        web_combined = web_reconciled.canonical_combined_artist
        web_title = web_reconciled.canonical_album_title
        web_artist_names: list[str] = (
            web_reconciled.canonical_artists or [pending.combined_artist]
        )

        web_album_text = compute_album_text(web_combined, web_title)
        web_artist_texts = [compute_artist_text(name) for name in web_artist_names]
        web_all_texts = [web_album_text] + web_artist_texts
        web_vectors = embed_texts(web_all_texts)
        web_album_vec = web_vectors[0]
        web_artist_vecs = web_vectors[1:]

        web_fp = compute_album_fingerprint(web_combined, web_title)

        web_enriched_artists = _resolve_constituent_artists(
            pending_artists=pending.artists,
            canonical_artists=web_artist_names,
            artist_embeddings=web_artist_vecs,
            cascade_artist_matches=cascade_result.artist_matches,
            preserve_remaining_pending=True,
        )

        return EnrichedAlbumData(
            album_id=pending.id,
            combined_artist=web_combined,
            title=web_title,
            fingerprint=web_fp,
            discogs_master_id=None,
            discogs_release_id=None,
            discogs_status=target_status,
            embedding=web_album_vec,
            artists=web_enriched_artists,
            reasoning=web_reconciled.reasoning,
        )

    def commit_matched_enrichment(self, enriched: EnrichedAlbumData) -> Album:
        """Phase 3: Update Album and constituent Artists in a short write session."""
        with self.session_factory() as session:
            album = session.get(Album, enriched.album_id)
            if album is None:
                raise ValueError(f"Album {enriched.album_id} not found for enrichment commit")

            # Check if canonical update collides with an existing Album row
            colliding_album_conds = [Album.fingerprint == enriched.fingerprint]
            if enriched.discogs_master_id is not None:
                colliding_album_conds.append(
                    Album.discogs_master_id == enriched.discogs_master_id
                )
            if enriched.discogs_release_id is not None:
                colliding_album_conds.append(
                    Album.discogs_release_id == enriched.discogs_release_id
                )

            colliding_album = session.execute(
                select(Album).where(
                    Album.id != album.id,
                    or_(*colliding_album_conds),
                )
            ).scalars().first()

            if colliding_album is not None:
                surviving_album = colliding_album
                superseded_album = album

                # 1. Re-parent album_artists from superseded_album to surviving_album
                superseded_aa = session.execute(
                    select(AlbumArtist).where(
                        AlbumArtist.album_id == superseded_album.id
                    )
                ).scalars().all()
                for aa in superseded_aa:
                    aa_exists = session.execute(
                        select(AlbumArtist).where(
                            AlbumArtist.album_id == surviving_album.id,
                            AlbumArtist.artist_id == aa.artist_id,
                        )
                    ).scalars().first()
                    if aa_exists is None:
                        session.add(
                            AlbumArtist(
                                album_id=surviving_album.id,
                                artist_id=aa.artist_id,
                            )
                        )
                    session.delete(aa)
                session.flush()

                # 2. Re-parent child Release rows from superseded_album to surviving_album
                superseded_releases = session.execute(
                    select(Release).where(Release.album_id == superseded_album.id)
                ).scalars().all()

                surviving_releases = session.execute(
                    select(Release).where(Release.album_id == surviving_album.id)
                ).scalars().all()
                surviving_by_fp = {r.fingerprint: r for r in surviving_releases}

                for rel in superseded_releases:
                    new_fp = compute_release_fingerprint(
                        surviving_album.id,
                        cast("str | None", rel.variant),
                        cast("str | None", rel.retailer_name),
                    )
                    if new_fp in surviving_by_fp:
                        surviving_rel = surviving_by_fp[new_fp]
                        # Re-link submission_releases
                        sub_rels = session.execute(
                            select(SubmissionRelease).where(
                                SubmissionRelease.release_id == rel.id
                            )
                        ).scalars().all()
                        for sr in sub_rels:
                            sr_exists = session.execute(
                                select(SubmissionRelease).where(
                                    SubmissionRelease.submission_id == sr.submission_id,
                                    SubmissionRelease.release_id == surviving_rel.id,
                                )
                            ).scalars().first()
                            if sr_exists is None:
                                session.add(
                                    SubmissionRelease(
                                        submission_id=sr.submission_id,
                                        release_id=surviving_rel.id,
                                    )
                                )
                            session.delete(sr)
                        session.flush()
                        session.delete(rel)
                    else:
                        rel.album_id = surviving_album.id
                        rel.fingerprint = new_fp
                        surviving_by_fp[new_fp] = rel
                session.flush()

                # 3. Delete superseded album
                session.delete(superseded_album)
                session.flush()

                # 4. Update surviving album metadata
                surviving_album.combined_artist = enriched.combined_artist
                surviving_album.title = enriched.title
                surviving_album.fingerprint = enriched.fingerprint
                if enriched.discogs_master_id is not None:
                    surviving_album.discogs_master_id = enriched.discogs_master_id
                if enriched.discogs_release_id is not None:
                    surviving_album.discogs_release_id = enriched.discogs_release_id
                surviving_album.embedding = enriched.embedding
                surviving_album.discogs_status = enriched.discogs_status
                surviving_album.enrichment_attempts = 0
                surviving_album.enrichment_error = None
                surviving_album.last_enriched_at = func.now()

                target_album = surviving_album
            else:
                album.combined_artist = enriched.combined_artist
                album.title = enriched.title
                album.fingerprint = enriched.fingerprint
                album.discogs_master_id = enriched.discogs_master_id
                album.discogs_release_id = enriched.discogs_release_id
                album.embedding = enriched.embedding
                album.discogs_status = enriched.discogs_status
                album.enrichment_attempts = 0
                album.enrichment_error = None
                album.last_enriched_at = func.now()

                target_album = album

            for art_data in enriched.artists:
                if art_data.artist_id is not None:
                    artist = session.get(Artist, art_data.artist_id)
                    if artist is not None:
                        # Check if updating this artist causes collision with another artist row
                        colliding_conds = [Artist.name == art_data.name]
                        if art_data.discogs_artist_id is not None:
                            colliding_conds.append(
                                Artist.discogs_artist_id == art_data.discogs_artist_id
                            )
                        colliding_artist = session.execute(
                            select(Artist).where(
                                Artist.id != artist.id,
                                or_(*colliding_conds),
                            )
                        ).scalars().first()

                        if colliding_artist is not None:
                            # Re-parent all AlbumArtist rows referencing artist.id across every album to colliding_artist.id
                            old_links = session.execute(
                                select(AlbumArtist).where(
                                    AlbumArtist.artist_id == artist.id
                                )
                            ).scalars().all()
                            for link in old_links:
                                stmt = (
                                    pg_insert(AlbumArtist)
                                    .values(
                                        album_id=link.album_id,
                                        artist_id=colliding_artist.id,
                                    )
                                    .on_conflict_do_nothing()
                                )
                                session.execute(stmt)
                            session.flush()

                            session.execute(
                                delete(AlbumArtist).where(
                                    AlbumArtist.artist_id == artist.id
                                )
                            )
                            session.flush()

                            session.delete(artist)
                            session.flush()

                            # Update surviving artist
                            colliding_artist.name = art_data.name
                            if art_data.discogs_artist_id is not None:
                                colliding_artist.discogs_artist_id = (
                                    art_data.discogs_artist_id
                                )
                            colliding_artist.embedding = art_data.embedding
                        else:
                            artist.name = art_data.name
                            if art_data.discogs_artist_id is not None:
                                artist.discogs_artist_id = art_data.discogs_artist_id
                            artist.embedding = art_data.embedding
                else:
                    where_clauses = [Artist.name == art_data.name]
                    if art_data.discogs_artist_id is not None:
                        where_clauses.append(
                            Artist.discogs_artist_id == art_data.discogs_artist_id
                        )
                    existing_art = session.execute(
                        select(Artist).where(or_(*where_clauses))
                    ).scalars().first()
                    if existing_art is None:
                        new_art = Artist(
                            name=art_data.name,
                            discogs_artist_id=art_data.discogs_artist_id,
                            embedding=art_data.embedding,
                        )
                        session.add(new_art)
                        session.flush()
                        target_art_id = new_art.id
                    else:
                        existing_art.name = art_data.name
                        if art_data.discogs_artist_id is not None:
                            existing_art.discogs_artist_id = art_data.discogs_artist_id
                        existing_art.embedding = art_data.embedding
                        target_art_id = existing_art.id

                    link_exists = session.execute(
                        select(AlbumArtist).where(
                            AlbumArtist.album_id == target_album.id,
                            AlbumArtist.artist_id == target_art_id,
                        )
                    ).scalars().first()
                    if link_exists is None:
                        session.add(
                            AlbumArtist(album_id=target_album.id, artist_id=target_art_id)
                        )

            session.commit()
            session.refresh(target_album)
            session.expunge(target_album)
            return target_album

    commit_enrichment = commit_matched_enrichment

    def record_failure(
        self,
        album_id: uuid.UUID | str | Any,
        error: str | Exception,
    ) -> None:
        """Record an unhandled enrichment failure in a short write transaction."""
        record_enrichment_failure(
            session_factory=self.session_factory,
            album_id=album_id,
            error=error,
        )

    def fetch_sibling_releases(
        self, album_id: uuid.UUID | str | Any
    ) -> list[SiblingReleaseInfo]:
        """Fetch child releases of an album in a short read session."""
        uid = uuid.UUID(str(album_id)) if not isinstance(album_id, uuid.UUID) else album_id
        with self.session_factory() as session:
            stmt = select(Release).where(Release.album_id == uid).order_by(Release.created_at)
            releases = session.execute(stmt).scalars().all()
            return [
                SiblingReleaseInfo(
                    id=cast(uuid.UUID, r.id),
                    variant=cast("str | None", r.variant),
                    retailer_name=cast("str | None", r.retailer_name),
                )
                for r in releases
            ]

    def commit_sibling_reconciliation(
        self,
        album_id: uuid.UUID | str | Any,
        duplicate_groups: Sequence[SiblingDuplicateGroup],
    ) -> int:
        """Write Tx 2: Merge duplicate sibling Release rows, re-link submission_releases, and delete duplicates."""
        if not duplicate_groups:
            return 0

        uid = uuid.UUID(str(album_id)) if not isinstance(album_id, uuid.UUID) else album_id
        deleted_count = 0

        with self.session_factory() as session:
            for group in duplicate_groups:
                canon_uid = (
                    uuid.UUID(str(group.canonical_release_id))
                    if not isinstance(group.canonical_release_id, uuid.UUID)
                    else group.canonical_release_id
                )
                canonical_rel = session.get(Release, canon_uid)
                if canonical_rel is None:
                    continue

                if group.standardized_variant is not None:
                    canonical_rel.variant = group.standardized_variant
                if group.standardized_retailer is not None:
                    canonical_rel.retailer_name = group.standardized_retailer
                canonical_rel.fingerprint = compute_release_fingerprint(
                    uid,
                    cast("str | None", canonical_rel.variant),
                    cast("str | None", canonical_rel.retailer_name),
                )

                for dup_id in group.duplicate_release_ids:
                    dup_uid = (
                        uuid.UUID(str(dup_id))
                        if not isinstance(dup_id, uuid.UUID)
                        else dup_id
                    )
                    dup_rel = session.get(Release, dup_uid)
                    if dup_rel is None or dup_rel.id == canonical_rel.id:
                        continue

                    # Re-link submission_releases
                    sub_rels = session.execute(
                        select(SubmissionRelease).where(
                            SubmissionRelease.release_id == dup_rel.id
                        )
                    ).scalars().all()
                    for sr in sub_rels:
                        sr_exists = session.execute(
                            select(SubmissionRelease).where(
                                SubmissionRelease.submission_id == sr.submission_id,
                                SubmissionRelease.release_id == canonical_rel.id,
                            )
                        ).scalars().first()
                        if sr_exists is None:
                            session.add(
                                SubmissionRelease(
                                    submission_id=sr.submission_id,
                                    release_id=canonical_rel.id,
                                )
                            )
                        session.delete(sr)
                    session.flush()
                    session.delete(dup_rel)
                    session.flush()
                    deleted_count += 1

            session.commit()
        return deleted_count

    def enrich_album(
        self,
        album_id: uuid.UUID | str | Any,
        raise_on_error: bool = False,
    ) -> Album | None:
        """Execute Phase 1, Phase 2, Phase 3, and Phase 4 detached enrichment for a single album."""
        pending = self.fetch_pending_album(album_id)
        if pending is None:
            return None

        try:
            enriched = self.execute_detached_enrichment(pending)
            if enriched is None:
                return None

            # Phase 3 (Short Write Tx 1)
            surviving_album = self.commit_matched_enrichment(enriched)

            # Phase 4 (Detached Sibling Release RAG Reconciliation + Short Write Tx 2)
            siblings = self.fetch_sibling_releases(surviving_album.id)
            if len(siblings) >= 2:
                reconciled = reconcile_sibling_releases(
                    combined_artist=str(surviving_album.combined_artist),
                    album_title=str(surviving_album.title),
                    siblings=siblings,
                    sibling_agent=self.sibling_agent,
                )
                if reconciled.duplicate_groups:
                    self.commit_sibling_reconciliation(
                        surviving_album.id, reconciled.duplicate_groups
                    )

            return surviving_album
        except Exception as exc:
            logger.exception("Enrichment failed for album %s", album_id)
            self.record_failure(album_id, exc)
            if raise_on_error:
                raise
            return None

    def process_pending_enrichment(
        self,
        limit: int | None = None,
        shutdown_event: Any = None,
        now: datetime | None = None,
    ) -> int:
        """Process queued albums through the 4-phase enrichment pipeline until queue empty or limit reached."""
        enriched_count = 0
        while True:
            if shutdown_event is not None and getattr(shutdown_event, "is_set", lambda: False)():
                break
            if limit is not None and enriched_count >= limit:
                break

            pending = self.fetch_next_queued_album(now=now)
            if pending is None:
                break

            enriched_album = self.enrich_album(pending.id)
            if enriched_album is not None and enriched_album.discogs_status in (
                "matched",
                "partial_matched",
                "not_found",
            ):
                enriched_count += 1

        return enriched_count


def process_pending_enrichment(
    session_factory: sessionmaker[Session] | Any,
    discogs_client: DiscogsClient,
    limit: int | None = None,
    shutdown_event: Any = None,
    **kwargs: Any,
) -> int:
    """Convenience helper to process pending albums for enrichment."""
    service = CatalogEnrichmentService(
        session_factory=session_factory,
        discogs_client=discogs_client,
        **kwargs,
    )
    return service.process_pending_enrichment(limit=limit, shutdown_event=shutdown_event)


def record_enrichment_failure(
    session_factory: sessionmaker[Session] | Any,
    album_id: uuid.UUID | str | Any,
    error: str | Exception,
) -> None:
    """Record an unhandled enrichment failure in a short write transaction."""
    uid = uuid.UUID(str(album_id)) if not isinstance(album_id, uuid.UUID) else album_id
    error_msg = str(error)
    with session_factory() as session:
        album = session.get(Album, uid)
        if album is not None:
            album.discogs_status = "failed"
            album.enrichment_attempts = (album.enrichment_attempts or 0) + 1
            album.last_enriched_at = func.now()
            album.enrichment_error = error_msg
            session.commit()


def enrich_album(
    album_id: uuid.UUID | str | Any,
    session_factory: sessionmaker[Session] | Any,
    discogs_client: DiscogsClient,
    arbiter_agent: Agent[None, CandidateArbiterResult] | None = None,
    disambiguation_agent: Agent[None, WorkReconciliationResult] | None = None,
    web_agent: Agent[None, WebReconciliationResult] | None = None,
    sibling_agent: Agent[None, SiblingReconciliationResult] | None = None,
    deepseek_api_key: str | None = None,
    deepseek_base_url: str | None = None,
    deepseek_http_client: httpx.Client | None = None,
    raise_on_error: bool = False,
) -> Album | None:
    """Convenience helper to enrich a single album using CatalogEnrichmentService."""
    service = CatalogEnrichmentService(
        session_factory=session_factory,
        discogs_client=discogs_client,
        arbiter_agent=arbiter_agent,
        disambiguation_agent=disambiguation_agent,
        web_agent=web_agent,
        sibling_agent=sibling_agent,
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        deepseek_http_client=deepseek_http_client,
    )
    return service.enrich_album(album_id, raise_on_error=raise_on_error)


__all__ = [
    "CANDIDATE_ARBITER_SYSTEM_PROMPT",
    "HOMONYM_DISAMBIGUATION_SYSTEM_PROMPT",
    "SIBLING_RECONCILIATION_SYSTEM_PROMPT",
    "WEB_RECONCILIATION_SYSTEM_PROMPT",
    "CandidateArbiterResult",
    "CanonicalArtistCredit",
    "CascadeResult",
    "CatalogEnrichmentService",
    "ConstituentArtistInfo",
    "EnrichedAlbumData",
    "EnrichedArtistData",
    "HydratedWorkDetail",
    "PendingAlbumData",
    "PendingConstituentArtist",
    "ResolvedArtistMatch",
    "ResolvedWorkMatch",
    "ScoredCandidate",
    "SiblingDuplicateGroup",
    "SiblingReconciliationResult",
    "SiblingReleaseInfo",
    "WebReconciliationResult",
    "WebSearchHit",
    "WorkReconciliationResult",
    "arbitrate_artist_candidates",
    "arbitrate_work_candidates",
    "build_artist_arbiter_prompt",
    "build_homonym_disambiguation_prompt",
    "build_sibling_reconciliation_prompt",
    "build_web_reconciliation_prompt",
    "build_work_arbiter_prompt",
    "compute_cosine_similarity",
    "create_candidate_arbiter_agent",
    "create_homonym_disambiguation_agent",
    "create_sibling_reconciliation_agent",
    "create_web_reconciliation_agent",
    "deduplicate_candidates",
    "enrich_album",
    "get_next_queued_album",
    "hydrate_work_detail",
    "process_pending_enrichment",
    "reconcile_artist_homonyms",
    "reconcile_sibling_releases",
    "reconcile_web_search",
    "reconstruct_combined_artist",
    "record_enrichment_failure",
    "run_discogs_search_cascade",
    "score_candidates",
    "search_web_for_release",
]




