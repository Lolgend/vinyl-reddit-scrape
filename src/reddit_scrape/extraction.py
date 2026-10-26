"""Structured entity extraction worker using DeepSeek LLM via PydanticAI."""

from __future__ import annotations

import inspect
import logging
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

if TYPE_CHECKING:
    from reddit_scrape.persistence import CatalogCandidates

CandidateRetriever = Callable[
    ...,
    Union[
        "CatalogCandidates",
        Awaitable["CatalogCandidates"],
        None,
    ],
]

load_dotenv()

logger = logging.getLogger(__name__)

# --- Thinking Level Logic ---


def get_default_thinking_level(model_name: str | None = None) -> str:
    """Return default thinking level based on model capabilities.

    If model_name is reasoning-capable (e.g. contains 'reasoner', 'r1', 'o1', 'o3'),
    returns 'low'. For standard chat models (e.g. 'deepseek-chat', 'gpt-4o'), returns 'none'.
    """
    model = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    model_lower = model.lower()
    reasoning_identifiers = ("reasoner", "r1", "o1", "o3")
    if any(ident in model_lower for ident in reasoning_identifiers):
        return "low"
    return "none"


# --- Schema Definitions ---


class ExtractedRelease(BaseModel):
    """Structured entity representing an individual release within a submission."""

    artists: list[str] = Field(
        description="Individual artist or band names contributing to the release (e.g. ['Converge', 'Dropdead'] or ['Tyler, The Creator'])"
    )
    combined_artist: str | None = Field(
        None,
        description="Combined billing string for multi-artist/collaborative releases (e.g. 'Converge / Dropdead', 'Kanye West & Ty Dolla $ign')",
    )
    album_title: str = Field(
        description="Album, EP, or single title"
    )
    variant: str | None = Field(
        None,
        description="Physical pressing variation (e.g. 'Opaque Emerald LP', '180g 45RPM 2LP Box', 'Pink /500')",
    )
    retailer: str | None = Field(
        None,
        description="Exclusive retailer, store, or record label if mentioned (e.g. 'Rough Trade', 'Blood Records', 'Target')",
    )


class ExtractionResult(BaseModel):
    """Structured output containing all releases extracted from a submission."""

    releases: list[ExtractedRelease] = Field(
        default_factory=list,
        description="List of distinct releases announced with specific album titles. Empty if non-release or generic store restock.",
    )


class BatchExtractedItem(BaseModel):
    """Structured entity representing releases extracted for a single submission in a batch."""

    submission_id: str = Field(
        description="Reddit submission ID corresponding to the submission."
    )
    releases: list[ExtractedRelease] = Field(
        default_factory=list,
        description="List of distinct releases announced with specific album titles. Empty if non-release or generic store restock.",
    )


class BatchExtractionResult(BaseModel):
    """Structured output containing extracted items for all submissions in a batch."""

    items: list[BatchExtractedItem] = Field(
        default_factory=list,
        description="List of extracted release items for each submission in the batch.",
    )


# --- Pre-cleaning Logic ---


def pre_clean_title(raw_title: str) -> str:
    """Normalize unicode dashes and collapse whitespace in raw submission titles."""
    if not raw_title:
        return ""
    cleaned = raw_title.strip()
    # Normalize unicode dashes: \u2010-\u2015 and \u2212 to standard ASCII hyphen
    cleaned = re.sub(r"[\u2010-\u2015\u2212]", "-", cleaned)
    # Collapse multiple whitespace characters into a single space
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# --- System Prompts ---

EXTRACTION_SYSTEM_PROMPT = """You are an expert vinyl record discography classifier specialized in extracting structured data from r/vinylreleases submission titles.

Rules:
1. Non-Release / Generic Submissions -> Return empty `releases: []`:
   - Community discussions, questions, or polls (e.g. "[Discussion] ...")
   - In-Search-Of requests (e.g. "[ISO] ...")
   - Store sales or generic restock announcements that do NOT name specific album titles (e.g. "VMP Restocks: Artist A, Artist B", "Target Buy 2 Get 1 Free Sale")
   
2. Specific Release Announcements -> Extract each release:
   - `artists`: List of individual artist/band names. For splits ('A / B') or collaborations ('A & B', 'A feat. B'), extract each artist as a separate list item. Keep multi-word/comma artist names intact (e.g. ["Tyler, The Creator"]).
   - `combined_artist`: For multi-artist releases, provide the collaborative billing string (e.g. "Converge / Dropdead", "Kanye West & Ty Dolla $ign"). For single artist releases, provide the single artist name.
   - `album_title`: Clean title of the album/single. Never hallucinate an album title if none is specified in the submission.
   - `variant`: The physical pressing variant details (e.g. "Opaque Emerald LP", "180g 45RPM 2LP Box", "Pink /500", "Clear Zoetrope").
   - `retailer`: Exclusive seller or store if mentioned (e.g. "Blood Records", "Target", "Rough Trade", "VMP").

3. Strict Anti-Hallucination:
   - Only extract information explicitly stated in the submission title. Do NOT guess or hallucinate missing album titles or artists.

4. Canonical Alignment (Existing Catalog Reference):
   When an "Existing Catalog Reference" section is present in the prompt:
   - Exact & Reordered Match Alignment: If the submission announces an artist or release that semantically matches an existing canonical record (even if artist/album order is reversed, multi-artist billing order is rearranged, or intra-field tokens differ in formatting/word order), you MUST align to the exact canonical strings from the reference for `combined_artist`, `artists`, `album_title`, `variant`, and `retailer`.
   - Partial Match Support: If the submission announces a new variant or retailer for an album already present in the catalog reference, reuse the exact canonical `artists`, `combined_artist`, and `album_title` from the reference while extracting the new `variant` and/or `retailer` from the submission.
   - Anti-Hallucination Preservation: Never force a match to an unrelated catalog candidate if the submission announces a genuinely different artist, album, or entity. Only align when the entity is semantically the same.
"""

BATCH_EXTRACTION_SYSTEM_PROMPT = """You are an expert vinyl record discography classifier specialized in extracting structured data from batches of r/vinylreleases submission titles.

Batch Processing Instructions:
- You will be given a batch of Reddit submissions, each with a Submission ID, optional Flair, and Title.
- Extract release entities for each submission in the batch.
- You MUST return exactly one `BatchExtractedItem` per provided `submission_id`, preserving the exact `submission_id`.

Rules for Each Submission:
1. Non-Release / Generic Submissions -> Return empty `releases: []`:
   - Community discussions, questions, or polls (e.g. "[Discussion] ...")
   - In-Search-Of requests (e.g. "[ISO] ...")
   - Store sales or generic restock announcements that do NOT name specific album titles (e.g. "VMP Restocks: Artist A, Artist B", "Target Buy 2 Get 1 Free Sale")

2. Specific Release Announcements -> Extract each release:
   - `artists`: List of individual artist/band names. For splits ('A / B') or collaborations ('A & B', 'A feat. B'), extract each artist as a separate list item. Keep multi-word/comma artist names intact (e.g. ["Tyler, The Creator"]).
   - `combined_artist`: For multi-artist releases, provide the collaborative billing string (e.g. "Converge / Dropdead", "Kanye West & Ty Dolla $ign"). For single artist releases, provide the single artist name.
   - `album_title`: Clean title of the album/single. Never hallucinate an album title if none is specified in the submission.
   - `variant`: The physical pressing variant details (e.g. "Opaque Emerald LP", "180g 45RPM 2LP Box", "Pink /500", "Clear Zoetrope").
   - `retailer`: Exclusive seller or store if mentioned (e.g. "Blood Records", "Target", "Rough Trade", "VMP").

3. Strict Anti-Hallucination:
   - Only extract information explicitly stated in the submission title. Do NOT guess or hallucinate missing album titles or artists.

4. Canonical Alignment (Existing Catalog Reference):
   When an "Existing Catalog Reference" section is present in the prompt:
   - Exact & Reordered Match Alignment: If the submission announces an artist or release that semantically matches an existing canonical record (even if artist/album order is reversed, multi-artist billing order is rearranged, or intra-field tokens differ in formatting/word order), you MUST align to the exact canonical strings from the reference for `combined_artist`, `artists`, `album_title`, `variant`, and `retailer`.
   - Partial Match Support: If the submission announces a new variant or retailer for an album already present in the catalog reference, reuse the exact canonical `artists`, `combined_artist`, and `album_title` from the reference while extracting the new `variant` and/or `retailer` from the submission.
   - Anti-Hallucination Preservation: Never force a match to an unrelated catalog candidate if the submission announces a genuinely different artist, album, or entity. Only align when the entity is semantically the same.

5. Intra-Batch Canonical Alignment:
   When multiple submissions within the same batch refer to the same Artist, Combined Artist, album, or Variant in different orders, arrangements, or word orders (e.g. "Converge / Dropdead" in one submission and "Dropdead / Converge" in another, or "Album - Artist" vs "Artist - Album"):
   - Output a single consistent canonical ordering for all of them across the entire batch.
   - If matching records exist in the "Existing Catalog Reference", all submissions in the batch MUST align to the reference canonical strings.
   - If no matching catalog reference exists, establish a consistent canonical order and billing string and reuse that exact same representation for all occurrences of that entity across the entire batch.
"""


# --- Prompt Builders ---


def format_catalog_reference(candidates: CatalogCandidates | None) -> str:
    """Format catalog candidates into an 'Existing Catalog Reference' block.

    Returns an empty string if candidates is None or has no artists or releases.
    """
    if not candidates:
        return ""
    if getattr(candidates, "is_empty", False):
        return ""

    has_artists = bool(getattr(candidates, "artists", None))
    has_releases = bool(getattr(candidates, "releases", None))

    if not has_artists and not has_releases:
        return ""

    lines = ["Existing Catalog Reference:"]
    if has_artists:
        lines.append("Canonical Artists:")
        for artist in candidates.artists:
            lines.append(f"- {artist}")

    if has_releases:
        if has_artists:
            lines.append("")
        lines.append("Canonical Releases:")
        for r in candidates.releases:
            artists_str = ", ".join(r.artists) if r.artists else r.combined_artist
            variant_str = r.variant if r.variant is not None else "None"
            retailer_str = r.retailer if r.retailer is not None else "None"
            lines.append(
                f"- Combined Artist: {r.combined_artist} | Artists: {artists_str} | Album Title: {r.album_title} | Variant: {variant_str} | Retailer: {retailer_str}"
            )

    return "\n".join(lines)


def build_extraction_prompt(
    title: str,
    flair: str | None = None,
    candidates: CatalogCandidates | None = None,
) -> str:
    """Build the LLM user prompt from the submission title, optional flair, and catalog candidates."""
    cleaned_title = pre_clean_title(title)
    if flair and flair.strip():
        base_prompt = (
            f"Extract release entities from Reddit submission:\n\n"
            f"Flair: {flair.strip()}\n"
            f"Title: {cleaned_title}"
        )
    else:
        base_prompt = f"Extract release entities from Reddit submission title:\n\n{cleaned_title}"

    reference_block = format_catalog_reference(candidates)
    if reference_block:
        return f"{base_prompt}\n\n{reference_block}"
    return base_prompt


def build_batch_extraction_prompt(
    submissions: Sequence[tuple[str, str, str | None]],
    candidates: CatalogCandidates | None = None,
) -> str:
    """Build the LLM user prompt for a batch of Reddit submissions."""
    if not submissions:
        return ""

    entries: list[str] = []
    for idx, (sub_id, title, flair) in enumerate(submissions, start=1):
        cleaned_title = pre_clean_title(title)
        entry_lines = [
            f"--- Submission {idx} ---",
            f"Submission ID: {sub_id}",
        ]
        if flair and flair.strip():
            entry_lines.append(f"Flair: {flair.strip()}")
        entry_lines.append(f"Title: {cleaned_title}")
        entries.append("\n".join(entry_lines))

    base_prompt = (
        "Extract release entities from the following Reddit submissions:\n\n"
        + "\n\n".join(entries)
    )

    reference_block = format_catalog_reference(candidates)
    if reference_block:
        return f"{base_prompt}\n\n{reference_block}"
    return base_prompt


# --- Agent Factories ---


def create_extraction_agent(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    thinking_level: str | None = None,
) -> Agent[None, ExtractionResult]:
    """Create a PydanticAI Agent configured for DeepSeek chat."""
    key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
    if not key:
        raise ValueError(
            "DEEPSEEK_API_KEY or DEEPSEEK_KEY environment variable is required"
        )
    url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    level = thinking_level or os.getenv("LLM_THINKING_LEVEL")
    if not level:
        level = get_default_thinking_level(model)

    model_settings: ModelSettings
    if level.lower() == "none":
        model_settings = {"thinking": False}
    else:
        model_settings = {"thinking": level.lower()}

    provider = OpenAIProvider(
        base_url=url,
        api_key=key,
    )
    chat_model = OpenAIChatModel(model, provider=provider)

    return Agent(
        chat_model,
        output_type=ExtractionResult,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        model_settings=model_settings,
    )


def create_batch_extraction_agent(
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str | None = None,
    thinking_level: str | None = None,
) -> Agent[None, BatchExtractionResult]:
    """Create a PydanticAI Agent configured for batch release extraction."""
    key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
    if not key:
        raise ValueError(
            "DEEPSEEK_API_KEY or DEEPSEEK_KEY environment variable is required"
        )
    url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    level = thinking_level or os.getenv("LLM_THINKING_LEVEL")
    if not level:
        level = get_default_thinking_level(model)

    model_settings: ModelSettings
    if level.lower() == "none":
        model_settings = {"thinking": False}
    else:
        model_settings = {"thinking": level.lower()}

    provider = OpenAIProvider(
        base_url=url,
        api_key=key,
    )
    chat_model = OpenAIChatModel(model, provider=provider)

    return Agent(
        chat_model,
        output_type=BatchExtractionResult,
        system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        model_settings=model_settings,
    )


# --- Asynchronous Extraction Functions ---


async def _resolve_candidates(
    retriever: CandidateRetriever | None,
    titles: Sequence[str],
) -> CatalogCandidates | None:
    """Safely execute a synchronous or asynchronous candidate retriever callable."""
    if retriever is None:
        return None
    try:
        try:
            result = retriever(titles)
        except TypeError:
            if len(titles) == 1:
                result = retriever(titles[0])
            else:
                raise
        if inspect.isawaitable(result):
            resolved = await result
        else:
            resolved = result
        if resolved is not None and hasattr(resolved, "releases") and hasattr(resolved, "artists"):
            return resolved  # type: ignore[return-value]
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Candidate retrieval failed (%s: %s); proceeding without catalog reference",
            type(exc).__name__,
            exc,
        )
        return None


async def extract_releases(
    title: str,
    flair: str | None = None,
    agent: Agent | None = None,
    candidates: CatalogCandidates | None = None,
    candidate_retriever: CandidateRetriever | None = None,
) -> ExtractionResult:
    """Extract structured releases from a submission title and optional flair asynchronously."""
    if agent is None:
        agent = create_extraction_agent()

    resolved_candidates = candidates
    if resolved_candidates is None and candidate_retriever is not None:
        resolved_candidates = await _resolve_candidates(candidate_retriever, [title])

    prompt = build_extraction_prompt(title, flair=flair, candidates=resolved_candidates)
    run_result = await agent.run(prompt)
    return run_result.output


class ExtractionWorker:
    """Worker encapsulating release extraction logic and agent lifecycle."""

    def __init__(
        self,
        agent: Agent | None = None,
        thinking_level: str | None = None,
        batch_agent: Agent | None = None,
        candidate_retriever: CandidateRetriever | None = None,
    ) -> None:
        self.agent = agent
        self.batch_agent = batch_agent
        self.thinking_level = thinking_level
        self.candidate_retriever = candidate_retriever
        self._owns_agent = agent is None
        self._owns_batch_agent = batch_agent is None

    def _get_agent(self) -> Agent:
        if self.agent is None:
            self.agent = create_extraction_agent(thinking_level=self.thinking_level)
        return self.agent

    def _get_batch_agent(self) -> Agent:
        if self.batch_agent is None:
            self.batch_agent = create_batch_extraction_agent(
                thinking_level=self.thinking_level
            )
        return self.batch_agent

    async def extract(
        self,
        title: str,
        flair: str | None = None,
        candidates: CatalogCandidates | None = None,
        candidate_retriever: CandidateRetriever | None = None,
    ) -> ExtractionResult:
        """Extract structured releases from a submission title and optional flair."""
        agent = self._get_agent()
        retriever = (
            candidate_retriever
            if candidate_retriever is not None
            else self.candidate_retriever
        )
        resolved_candidates = candidates
        if resolved_candidates is None and retriever is not None:
            resolved_candidates = await _resolve_candidates(retriever, [title])

        kwargs: dict[str, Any] = {}
        if resolved_candidates is not None:
            kwargs["candidates"] = resolved_candidates

        try:
            return await extract_releases(
                title,
                flair=flair,
                agent=agent,
                **kwargs,
            )
        except RuntimeError as exc:
            if "Event loop is closed" in str(exc) and self._owns_agent:
                self.agent = create_extraction_agent(thinking_level=self.thinking_level)
                return await extract_releases(
                    title,
                    flair=flair,
                    agent=self.agent,
                    **kwargs,
                )
            raise

    async def extract_batch(
        self,
        submissions: Sequence[tuple[str, str, str | None]],
        batch_size: int = 5,
        candidates: CatalogCandidates | None = None,
        candidate_retriever: CandidateRetriever | None = None,
    ) -> dict[str, ExtractionResult]:
        """Extract releases for a batch of submissions with automatic single-submission fallback."""
        if not submissions:
            return {}

        effective_batch_size = max(1, batch_size)
        if len(submissions) > effective_batch_size:
            combined_results: dict[str, ExtractionResult] = {}
            for i in range(0, len(submissions), effective_batch_size):
                sub_batch = submissions[i : i + effective_batch_size]
                sub_results = await self.extract_batch(
                    sub_batch,
                    batch_size=effective_batch_size,
                    candidates=candidates,
                    candidate_retriever=candidate_retriever,
                )
                combined_results.update(sub_results)
            return combined_results

        retriever = (
            candidate_retriever
            if candidate_retriever is not None
            else self.candidate_retriever
        )
        resolved_candidates = candidates
        if resolved_candidates is None and retriever is not None:
            chunk_titles = [title for _, title, _ in submissions]
            resolved_candidates = await _resolve_candidates(retriever, chunk_titles)

        prompt = build_batch_extraction_prompt(submissions, candidates=resolved_candidates)
        agent = self._get_batch_agent()
        sub_map = {sub_id: (title, flair) for sub_id, title, flair in submissions}

        try:
            try:
                run_result = await agent.run(prompt)
            except RuntimeError as exc:
                if "Event loop is closed" in str(exc) and self._owns_batch_agent:
                    self.batch_agent = create_batch_extraction_agent(
                        thinking_level=self.thinking_level
                    )
                    run_result = await self.batch_agent.run(prompt)
                else:
                    raise

            results: dict[str, ExtractionResult] = {
                item.submission_id: ExtractionResult(releases=item.releases)
                for item in run_result.output.items
                if item.submission_id in sub_map
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Batch extraction failed (%s: %s); falling back to single-submission extraction for %d items",
                type(exc).__name__,
                exc,
                len(submissions),
            )
            fallback_results: dict[str, ExtractionResult] = {}
            fallback_candidates = candidates if candidates is not None else resolved_candidates
            for sub_id, title, flair in submissions:
                try:
                    try:
                        fallback_results[sub_id] = await self.extract(
                            title,
                            flair=flair,
                            candidates=fallback_candidates,
                            candidate_retriever=retriever,
                        )
                    except TypeError:
                        fallback_results[sub_id] = await self.extract(title, flair=flair)
                except Exception as single_exc:  # noqa: BLE001
                    logger.warning(
                        "Fallback extraction failed for submission %s (%s: %s)",
                        sub_id,
                        type(single_exc).__name__,
                        single_exc,
                    )
                    fallback_results[sub_id] = ExtractionResult(releases=[])
            return fallback_results

        missing_ids = [sub_id for sub_id in sub_map if sub_id not in results]
        if missing_ids:
            logger.warning(
                "Batch extraction omitted %d submission IDs; falling back to single-submission extraction for omitted items",
                len(missing_ids),
            )
            fallback_candidates = candidates if candidates is not None else resolved_candidates
            for sub_id in missing_ids:
                title, flair = sub_map[sub_id]
                try:
                    try:
                        results[sub_id] = await self.extract(
                            title,
                            flair=flair,
                            candidates=fallback_candidates,
                            candidate_retriever=retriever,
                        )
                    except TypeError:
                        results[sub_id] = await self.extract(title, flair=flair)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Fallback extraction failed for omitted submission %s (%s: %s)",
                        sub_id,
                        type(exc).__name__,
                        exc,
                    )
                    results[sub_id] = ExtractionResult(releases=[])

        return results


__all__ = [
    "BATCH_EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_SYSTEM_PROMPT",
    "BatchExtractedItem",
    "BatchExtractionResult",
    "CandidateRetriever",
    "ExtractedRelease",
    "ExtractionResult",
    "ExtractionWorker",
    "build_batch_extraction_prompt",
    "build_extraction_prompt",
    "create_batch_extraction_agent",
    "create_extraction_agent",
    "extract_releases",
    "format_catalog_reference",
    "get_default_thinking_level",
    "pre_clean_title",
]
