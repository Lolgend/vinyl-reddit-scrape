"""Structured entity extraction worker using DeepSeek LLM via PydanticAI."""

from __future__ import annotations

import os
import re
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

# --- Schema Definitions ---

class ExtractedRelease(BaseModel):
    """Structured entity representing an individual release within a submission."""

    artists: list[str] = Field(
        description="Individual artist or band names contributing to the release (e.g. ['Converge', 'Dropdead'] or ['Tyler, The Creator'])"
    )
    combined_artist: Optional[str] = Field(
        None,
        description="Combined billing string for multi-artist/collaborative releases (e.g. 'Converge / Dropdead', 'Kanye West & Ty Dolla $ign')",
    )
    album_title: str = Field(
        description="Album, EP, or single title"
    )
    variant: Optional[str] = Field(
        None,
        description="Physical pressing variation (e.g. 'Opaque Emerald LP', '180g 45RPM 2LP Box', 'Pink /500')",
    )
    retailer: Optional[str] = Field(
        None,
        description="Exclusive retailer, store, or record label if mentioned (e.g. 'Rough Trade', 'Blood Records', 'Target')",
    )


class ExtractionResult(BaseModel):
    """Structured output containing all releases extracted from a submission."""

    releases: list[ExtractedRelease] = Field(
        default_factory=list,
        description="List of distinct releases announced with specific album titles. Empty if non-release or generic store restock.",
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


# --- System Prompt ---

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


# --- Prompt Builder ---

def build_extraction_prompt(title: str, flair: Optional[str] = None) -> str:
    """Build the LLM user prompt from the submission title and optional flair."""
    cleaned_title = pre_clean_title(title)
    if flair and flair.strip():
        return (
            f"Extract release entities from Reddit post:\n\n"
            f"Flair: {flair.strip()}\n"
            f"Title: {cleaned_title}"
        )
    return f"Extract release entities from Reddit post title:\n\n{cleaned_title}"


# --- Agent Factory ---

def create_extraction_agent(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Agent[None, ExtractionResult]:
    """Create a PydanticAI Agent configured for DeepSeek chat."""
    key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY")
    if not key:
        raise ValueError(
            "DEEPSEEK_API_KEY or DEEPSEEK_KEY environment variable is required"
        )
    url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = model_name or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    provider = OpenAIProvider(
        base_url=url,
        api_key=key,
    )
    chat_model = OpenAIChatModel(model, provider=provider)

    return Agent(
        chat_model,
        output_type=ExtractionResult,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
    )


# --- Asynchronous Extraction Functions ---

async def extract_releases(
    title: str,
    flair: Optional[str] = None,
    agent: Optional[Agent] = None,
) -> ExtractionResult:
    """Extract structured releases from a post title and optional flair asynchronously."""
    if agent is None:
        agent = create_extraction_agent()

    prompt = build_extraction_prompt(title, flair)
    run_result = await agent.run(prompt)
    return run_result.output


class ExtractionWorker:
    """Worker encapsulating release extraction logic and agent lifecycle."""

    def __init__(self, agent: Optional[Agent] = None) -> None:
        self.agent = agent

    def _get_agent(self) -> Agent:
        if self.agent is None:
            self.agent = create_extraction_agent()
        return self.agent

    async def extract(
        self,
        title: str,
        flair: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract structured releases from a submission title and optional flair."""
        agent = self._get_agent()
        return await extract_releases(title, flair=flair, agent=agent)


__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "ExtractedRelease",
    "ExtractionResult",
    "ExtractionWorker",
    "build_extraction_prompt",
    "create_extraction_agent",
    "extract_releases",
    "pre_clean_title",
]
