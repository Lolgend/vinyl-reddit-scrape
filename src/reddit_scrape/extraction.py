"""Structured entity extraction worker using DeepSeek LLM via PydanticAI."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

load_dotenv()

logger = logging.getLogger(__name__)

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
        description="Reddit submission ID corresponding to the post."
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
1. Non-Release / Generic Posts -> Return empty `releases: []`:
   - Community discussions, questions, or polls (e.g. "[Discussion] ...")
   - In-Search-Of requests (e.g. "[ISO] ...")
   - Store sales or generic restock announcements that do NOT name specific album titles (e.g. "VMP Restocks: Artist A, Artist B", "Target Buy 2 Get 1 Free Sale")
   
2. Specific Release Announcements -> Extract each release:
   - `artists`: List of individual artist/band names. For splits ('A / B') or collaborations ('A & B', 'A feat. B'), extract each artist as a separate list item. Keep multi-word/comma artist names intact (e.g. ["Tyler, The Creator"]).
   - `combined_artist`: For multi-artist releases, provide the collaborative billing string (e.g. "Converge / Dropdead", "Kanye West & Ty Dolla $ign"). For single artist releases, provide the single artist name.
   - `album_title`: Clean title of the album/single. Never hallucinate an album title if none is specified in the post.
   - `variant`: The physical pressing variant details (e.g. "Opaque Emerald LP", "180g 45RPM 2LP Box", "Pink /500", "Clear Zoetrope").
   - `retailer`: Exclusive seller or store if mentioned (e.g. "Blood Records", "Target", "Rough Trade", "VMP").

3. Strict Anti-Hallucination:
   - Only extract information explicitly stated in the post title. Do NOT guess or hallucinate missing album titles or artists.
"""

BATCH_EXTRACTION_SYSTEM_PROMPT = """You are an expert vinyl record discography classifier specialized in extracting structured data from batches of r/vinylreleases submission titles.

Batch Processing Instructions:
- You will be given a batch of Reddit submissions, each with a Submission ID, optional Flair, and Title.
- Extract release entities for each submission in the batch.
- You MUST return exactly one `BatchExtractedItem` per provided `submission_id`, preserving the exact `submission_id`.

Rules for Each Submission:
1. Non-Release / Generic Posts -> Return empty `releases: []`:
   - Community discussions, questions, or polls (e.g. "[Discussion] ...")
   - In-Search-Of requests (e.g. "[ISO] ...")
   - Store sales or generic restock announcements that do NOT name specific album titles (e.g. "VMP Restocks: Artist A, Artist B", "Target Buy 2 Get 1 Free Sale")

2. Specific Release Announcements -> Extract each release:
   - `artists`: List of individual artist/band names. For splits ('A / B') or collaborations ('A & B', 'A feat. B'), extract each artist as a separate list item. Keep multi-word/comma artist names intact (e.g. ["Tyler, The Creator"]).
   - `combined_artist`: For multi-artist releases, provide the collaborative billing string (e.g. "Converge / Dropdead", "Kanye West & Ty Dolla $ign"). For single artist releases, provide the single artist name.
   - `album_title`: Clean title of the album/single. Never hallucinate an album title if none is specified in the post.
   - `variant`: The physical pressing variant details (e.g. "Opaque Emerald LP", "180g 45RPM 2LP Box", "Pink /500", "Clear Zoetrope").
   - `retailer`: Exclusive seller or store if mentioned (e.g. "Blood Records", "Target", "Rough Trade", "VMP").

3. Strict Anti-Hallucination:
   - Only extract information explicitly stated in the post title. Do NOT guess or hallucinate missing album titles or artists.
"""


# --- Prompt Builders ---


def build_extraction_prompt(title: str, flair: str | None = None) -> str:
    """Build the LLM user prompt from the submission title and optional flair."""
    cleaned_title = pre_clean_title(title)
    if flair and flair.strip():
        return (
            f"Extract release entities from Reddit post:\n\n"
            f"Flair: {flair.strip()}\n"
            f"Title: {cleaned_title}"
        )
    return f"Extract release entities from Reddit post title:\n\n{cleaned_title}"


def build_batch_extraction_prompt(
    submissions: Sequence[tuple[str, str, str | None]],
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

    return (
        "Extract release entities from the following Reddit submissions:\n\n"
        + "\n\n".join(entries)
    )


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

    level = thinking_level or os.getenv("LLM_THINKING_LEVEL", "low")
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

    level = thinking_level or os.getenv("LLM_THINKING_LEVEL", "low")
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


async def extract_releases(
    title: str,
    flair: str | None = None,
    agent: Agent | None = None,
) -> ExtractionResult:
    """Extract structured releases from a post title and optional flair asynchronously."""
    if agent is None:
        agent = create_extraction_agent()

    prompt = build_extraction_prompt(title, flair)
    run_result = await agent.run(prompt)
    return run_result.output


class ExtractionWorker:
    """Worker encapsulating release extraction logic and agent lifecycle."""

    def __init__(
        self,
        agent: Agent | None = None,
        thinking_level: str | None = None,
        batch_agent: Agent | None = None,
    ) -> None:
        self.agent = agent
        self.batch_agent = batch_agent
        self.thinking_level = thinking_level
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
    ) -> ExtractionResult:
        """Extract structured releases from a submission title and optional flair."""
        agent = self._get_agent()
        try:
            return await extract_releases(title, flair=flair, agent=agent)
        except RuntimeError as exc:
            if "Event loop is closed" in str(exc) and self._owns_agent:
                self.agent = create_extraction_agent(thinking_level=self.thinking_level)
                return await extract_releases(title, flair=flair, agent=self.agent)
            raise

    async def extract_batch(
        self,
        submissions: Sequence[tuple[str, str, str | None]],
    ) -> dict[str, ExtractionResult]:
        """Extract releases for a batch of submissions with automatic single-submission fallback."""
        if not submissions:
            return {}

        prompt = build_batch_extraction_prompt(submissions)
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
            for sub_id, title, flair in submissions:
                fallback_results[sub_id] = await self.extract(title, flair=flair)
            return fallback_results

        missing_ids = [sub_id for sub_id in sub_map if sub_id not in results]
        if missing_ids:
            logger.warning(
                "Batch extraction omitted %d submission IDs; falling back to single-submission extraction for omitted items",
                len(missing_ids),
            )
            for sub_id in missing_ids:
                title, flair = sub_map[sub_id]
                results[sub_id] = await self.extract(title, flair=flair)

        return results


__all__ = [
    "BATCH_EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_SYSTEM_PROMPT",
    "BatchExtractedItem",
    "BatchExtractionResult",
    "ExtractedRelease",
    "ExtractionResult",
    "ExtractionWorker",
    "build_batch_extraction_prompt",
    "build_extraction_prompt",
    "create_batch_extraction_agent",
    "create_extraction_agent",
    "extract_releases",
    "pre_clean_title",
]
