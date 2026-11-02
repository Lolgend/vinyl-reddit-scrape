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
from typing import Any, cast

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from sqlalchemy import and_, case, func, literal, or_, select, text
from sqlalchemy.orm import Session, sessionmaker

from reddit_scrape.discogs import (
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
from reddit_scrape.models import Album, AlbumArtist, Artist
from reddit_scrape.persistence import (
    compute_album_fingerprint,
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
    if model is not None:
        return Agent(
            model,
            output_type=WorkReconciliationResult,
            system_prompt=HOMONYM_DISAMBIGUATION_SYSTEM_PROMPT,
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
        output_type=WorkReconciliationResult,
        system_prompt=HOMONYM_DISAMBIGUATION_SYSTEM_PROMPT,
        model_settings=model_settings,
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
    if model is not None:
        return Agent(
            model,
            output_type=WebReconciliationResult,
            system_prompt=WEB_RECONCILIATION_SYSTEM_PROMPT,
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
        output_type=WebReconciliationResult,
        system_prompt=WEB_RECONCILIATION_SYSTEM_PROMPT,
        model_settings=model_settings,
    )


def reconcile_web_search(
    combined_artist: str,
    album_title: str,
    hits: Sequence[WebSearchHit],
    web_agent: Agent[None, WebReconciliationResult] | None = None,
) -> WebReconciliationResult:
    """Pass web search results through LLM to standardize canonical artist/album names.

    Gracefully falls back to extracted names if hits is empty or agent execution errors.
    """
    fallback = WebReconciliationResult(
        canonical_combined_artist=combined_artist,
        canonical_artists=[combined_artist],
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

    p0_cond = Album.discogs_status == "pending"

    p1_cond = and_(
        Album.discogs_status == "failed",
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

    p2_cond = and_(
        Album.discogs_status == "partial_matched",
        or_(
            Album.last_enriched_at.is_(None),
            Album.last_enriched_at <= now_expr - text("interval '7 days'"),
        ),
        Album.created_at >= now_expr - text("interval '60 days'"),
    )

    priority_case = case(
        (Album.discogs_status == "pending", 0),
        (Album.discogs_status == "failed", 1),
        (Album.discogs_status == "partial_matched", 2),
        else_=9,
    )

    stmt = (
        select(Album)
        .where(or_(p0_cond, p1_cond, p2_cond))
        .order_by(
            priority_case.asc(),
            Album.created_at.asc(),
            Album.last_enriched_at.asc().nulls_first(),
        )
        .limit(1)
    )

    return session.execute(stmt).scalars().first()


class CatalogEnrichmentService:
    """Orchestrates multi-phase detached Discogs catalog enrichment."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] | Any,
        discogs_client: DiscogsClient,
        arbiter_agent: Agent[None, CandidateArbiterResult] | None = None,
        disambiguation_agent: Agent[None, WorkReconciliationResult] | None = None,
        web_agent: Agent[None, WebReconciliationResult] | None = None,
        deepseek_api_key: str | None = None,
        deepseek_base_url: str | None = None,
        deepseek_http_client: httpx.Client | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.discogs_client = discogs_client
        self.arbiter_agent = arbiter_agent
        self.disambiguation_agent = disambiguation_agent
        self.web_agent = web_agent
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

            enriched_artists: list[EnrichedArtistData] = []
            remaining_pending = list(pending.artists)
            total_canonical = len(canonical_artists)

            for idx, (ca, vec) in enumerate(zip(canonical_artists, artist_vecs)):
                norm_ca = normalize_attribute(ca.name)
                norm_ca_clean = normalize_attribute(_clean_homonym_suffix(ca.name))

                # 1. Exact match on normalized name
                matched_pending = next(
                    (p for p in remaining_pending if normalize_attribute(p.name) == norm_ca),
                    None,
                )
                # 2. Match after stripping homonym suffixes (n)
                if matched_pending is None:
                    matched_pending = next(
                        (
                            p
                            for p in remaining_pending
                            if normalize_attribute(_clean_homonym_suffix(p.name)) == norm_ca_clean
                        ),
                        None,
                    )
                # 3. Match by position if remaining counts align
                if matched_pending is None and len(remaining_pending) == (total_canonical - idx) or matched_pending is None and len(remaining_pending) == 1:
                    matched_pending = remaining_pending[0]

                if matched_pending is not None:
                    remaining_pending.remove(matched_pending)
                    p_id = matched_pending.id
                else:
                    p_id = None

                enriched_artists.append(
                    EnrichedArtistData(
                        artist_id=p_id,
                        name=ca.name,
                        discogs_artist_id=ca.discogs_artist_id,
                        embedding=vec,
                    )
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

        web_enriched_artists: list[EnrichedArtistData] = []
        remaining_pending = list(pending.artists)
        total_canonical = len(web_artist_names)

        for idx, (art_name, vec) in enumerate(zip(web_artist_names, web_artist_vecs)):
            norm_ca = normalize_attribute(art_name)
            norm_ca_clean = normalize_attribute(_clean_homonym_suffix(art_name))

            matched_pending = next(
                (p for p in remaining_pending if normalize_attribute(p.name) == norm_ca),
                None,
            )
            if matched_pending is None:
                matched_pending = next(
                    (
                        p
                        for p in remaining_pending
                        if normalize_attribute(_clean_homonym_suffix(p.name)) == norm_ca_clean
                    ),
                    None,
                )
            if (
                matched_pending is None
                and len(remaining_pending) == (total_canonical - idx)
                or matched_pending is None
                and len(remaining_pending) == 1
            ):
                matched_pending = remaining_pending[0]

            if matched_pending is not None:
                remaining_pending.remove(matched_pending)
                p_id = matched_pending.id
                discogs_id = matched_pending.discogs_artist_id
                if discogs_id is None and matched_pending.name in cascade_result.artist_matches:
                    discogs_id = cascade_result.artist_matches[matched_pending.name].discogs_artist_id
            else:
                p_id = None
                discogs_id = None
                if art_name in cascade_result.artist_matches:
                    discogs_id = cascade_result.artist_matches[art_name].discogs_artist_id

            web_enriched_artists.append(
                EnrichedArtistData(
                    artist_id=p_id,
                    name=art_name,
                    discogs_artist_id=discogs_id,
                    embedding=vec,
                )
            )

        for p in remaining_pending:
            p_vec = embed_texts([compute_artist_text(p.name)])[0]
            discogs_id = p.discogs_artist_id
            if discogs_id is None and p.name in cascade_result.artist_matches:
                discogs_id = cascade_result.artist_matches[p.name].discogs_artist_id
            web_enriched_artists.append(
                EnrichedArtistData(
                    artist_id=p.id,
                    name=p.name,
                    discogs_artist_id=discogs_id,
                    embedding=p_vec,
                )
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

            for art_data in enriched.artists:
                if art_data.artist_id is not None:
                    artist = session.get(Artist, art_data.artist_id)
                    if artist is not None:
                        artist.name = art_data.name
                        if art_data.discogs_artist_id is not None:
                            artist.discogs_artist_id = art_data.discogs_artist_id
                        artist.embedding = art_data.embedding
                else:
                    where_clauses = [Artist.name == art_data.name]
                    if art_data.discogs_artist_id is not None:
                        where_clauses.append(Artist.discogs_artist_id == art_data.discogs_artist_id)
                    existing_art = session.execute(
                        select(Artist).where(
                            where_clauses[0] if len(where_clauses) == 1 else (where_clauses[0] | where_clauses[1])
                        )
                    ).scalar_one_or_none()
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
                            AlbumArtist.album_id == album.id,
                            AlbumArtist.artist_id == target_art_id,
                        )
                    ).scalar_one_or_none()
                    if link_exists is None:
                        session.add(AlbumArtist(album_id=album.id, artist_id=target_art_id))

            session.commit()
            session.refresh(album)
            session.expunge(album)
            return album

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

    def enrich_album(
        self,
        album_id: uuid.UUID | str | Any,
        raise_on_error: bool = False,
    ) -> Album | None:
        """Execute Phase 1, Phase 2, and Phase 3 detached enrichment for a single album."""
        pending = self.fetch_pending_album(album_id)
        if pending is None:
            return None

        try:
            enriched = self.execute_detached_enrichment(pending)
            if enriched is not None:
                return self.commit_matched_enrichment(enriched)
            return None
        except Exception as exc:
            logger.exception("Enrichment failed for album %s", album_id)
            self.record_failure(album_id, exc)
            if raise_on_error:
                raise
            return None


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
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=deepseek_base_url,
        deepseek_http_client=deepseek_http_client,
    )
    return service.enrich_album(album_id, raise_on_error=raise_on_error)


__all__ = [
    "CANDIDATE_ARBITER_SYSTEM_PROMPT",
    "HOMONYM_DISAMBIGUATION_SYSTEM_PROMPT",
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
    "WebReconciliationResult",
    "WebSearchHit",
    "WorkReconciliationResult",
    "arbitrate_artist_candidates",
    "arbitrate_work_candidates",
    "build_artist_arbiter_prompt",
    "build_homonym_disambiguation_prompt",
    "build_web_reconciliation_prompt",
    "build_work_arbiter_prompt",
    "compute_cosine_similarity",
    "create_candidate_arbiter_agent",
    "create_homonym_disambiguation_agent",
    "create_web_reconciliation_agent",
    "deduplicate_candidates",
    "enrich_album",
    "get_next_queued_album",
    "hydrate_work_detail",
    "reconcile_artist_homonyms",
    "reconcile_web_search",
    "reconstruct_combined_artist",
    "record_enrichment_failure",
    "run_discogs_search_cascade",
    "score_candidates",
    "search_web_for_release",
]




