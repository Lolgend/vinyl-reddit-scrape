"""
PROTOTYPE: DeepSeek PydanticAI Extraction Schema & Prompt (Issue #5 - Refined)
-----------------------------------------------------------------------------
Refined based on HITL review feedback:
1. Ignore non-related posts (discussions, ISO, non-specific restocks not mentioning albums).
2. Removed confidence, is_soundtrack, limitation_count, release_type.
3. Switched color_variant / format_variant to a unified `variant: Optional[str]`.
4. Multi-artist albums: tagged under individual `artists: list[str]` and `combined_artist: Optional[str]`.
"""

import json
import os
import re
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()


# --- Refined Domain Extraction Schemas ---

class ExtractedRelease(BaseModel):
    artists: list[str] = Field(
        description="List of individual contributing artist or band names (e.g. ['Converge', 'Dropdead'] or ['Kanye West', 'Ty Dolla $ign'] or ['Charli XCX'])"
    )
    combined_artist: Optional[str] = Field(
        None,
        description="Combined billing display string when multiple artists collaborate (e.g. 'Converge / Dropdead', 'Kanye West & Ty Dolla $ign'). For single artists, can be the artist name or null."
    )
    album_title: str = Field(
        description="Clean title of the album, EP, single, or box set without variant/date noise"
    )
    variant: Optional[str] = Field(
        None,
        description="Physical pressing variation (colorway, 180g, LP count, zoetrope, limitation text, etc.), e.g. 'Opaque Emerald LP', '180g 45RPM 2LP Box', 'Pink Swirl /500'"
    )
    retailer: Optional[str] = Field(
        None,
        description="Exclusive retailer, store, or record label (e.g., 'Rough Trade', 'Blood Records', 'Target', 'VMP')"
    )


class ExtractionResult(BaseModel):
    is_vinyl_release: bool = Field(
        description="Whether this submission announces a specific vinyl album release. FALSE for discussions, questions, ISO, gear, or general restocks/sales that do NOT name specific album titles."
    )
    releases: list[ExtractedRelease] = Field(
        default_factory=list,
        description="List of distinct releases announced with specific album titles. Empty if non-release or generic non-specific store restock."
    )
    reasoning: Optional[str] = Field(
        None,
        description="Brief explanation of extraction decisions and filtration"
    )


# --- Pre-cleaning Logic ---

def pre_clean_title(raw_title: str) -> str:
    """Normalize unicode dashes, whitespace, and clean noise."""
    cleaned = raw_title.strip()
    cleaned = re.sub(r"[\u2010-\u2015\u2212]", "-", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


# --- Agent Factory ---

SYSTEM_PROMPT = """You are an expert vinyl record discography classifier specialized in extracting structured data from r/vinylreleases submission titles.

Strict Ingestion Rules:
1. `is_vinyl_release`:
   - Set to FALSE and return empty `releases: []` for:
     * Community discussions, polls, or questions (e.g. "[Discussion] ...")
     * In-Search-Of requests (e.g. "[ISO] ...")
     * Equipment, turntable, setup, or care advice
     * Storefront sales or generic restock announcements that do NOT mention specific album titles (e.g. "VMP Restocks: Artist A, Artist B", "Target Buy 2 Get 1 Sale", "Assorted Blue Note Restocks")
   - Set to TRUE only if the post announces one or more specific vinyl album/EP/single releases with identifiable album titles.

2. Release Entities:
   - `artists`: List of individual artist/band names. For splits ('A / B'), collaborations ('A & B', 'A x B', 'A feat. B'), extract each artist as a separate element in the list. Keep multi-word/comma artist names intact (e.g. ["Tyler, The Creator"]).
   - `combined_artist`: For multi-artist releases, provide the full combined collaboration billing (e.g. "Converge / Dropdead", "Kanye West & Ty Dolla $ign"). For single artist, provide the artist name.
   - `album_title`: Clean title of the album/single. Do NOT hallucinate an album title if none is stated in the post. If no album title is specified, do not extract a release.
   - `variant`: The physical pressing variant details (e.g. "Opaque Emerald LP", "180g 45RPM 2LP Box", "Pink Target Exclusive", "Clear /500 Zoetrope").
   - `retailer`: Exclusive seller, store, or label if mentioned (e.g. "Blood Records", "Target", "Rough Trade", "VMP").

3. Strict Anti-Hallucination:
   - NEVER invent or guess album titles or artists not explicitly present in the post title.
"""

def create_extraction_agent(api_key: Optional[str] = None) -> Agent:
    key = api_key or os.getenv("DEEPSEEK_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_KEY or DEEPSEEK_API_KEY environment variable is required")
    
    provider = OpenAIProvider(
        base_url="https://api.deepseek.com",
        api_key=key,
    )
    model = OpenAIChatModel("deepseek-chat", provider=provider)
    
    agent = Agent(
        model,
        output_type=ExtractionResult,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


# --- Refined Benchmark Cases ---

BENCHMARK_CASES = [
    {
        "id": "canonical",
        "category": "Canonical Subreddit Rule #3",
        "title": "Fontaines D.C. - Romance (Opaque Emerald LP) [$32.99]",
        "expected_release": True,
        "desc": "Standard title with price to strip and variant to capture."
    },
    {
        "id": "preorder_variant",
        "category": "Pre-Order & Variant with Limitation",
        "title": "[Pre-Order] [10/24] Charli XCX - Brat (Translucent Black Ice Vinyl /2000) [Rough Trade]",
        "expected_release": True,
        "desc": "Variant captures translucent black ice and /2000 limitation text cleanly."
    },
    {
        "id": "audiophile_weight",
        "category": "Audiophile Weight Variant (180g / 45RPM)",
        "title": "Miles Davis – Kind of Blue (Mobile Fidelity 180g 45RPM 2LP Box)",
        "expected_release": True,
        "desc": "Tests non-color physical variant (180g, 45RPM, box)."
    },
    {
        "id": "multi_artist_split",
        "category": "Multi-Artist Split Release",
        "title": "Converge / Dropdead - Split (7\" /500)",
        "expected_release": True,
        "desc": "Split release: individual artists ['Converge', 'Dropdead'] and combined 'Converge / Dropdead'."
    },
    {
        "id": "multi_artist_collab",
        "category": "Multi-Artist Collaboration",
        "title": "Kanye West & Ty Dolla $ign - Vultures 1 (Black 2LP)",
        "expected_release": True,
        "desc": "Collaboration: individual artists ['Kanye West', 'Ty Dolla $ign'] and combined relation."
    },
    {
        "id": "comma_artist_restock",
        "category": "Internal Comma in Artist with Specific Album",
        "title": "RESTOCK: Tyler, The Creator - IGOR (Mint Green Special Edition LP)",
        "expected_release": True,
        "desc": "Valid restock because specific album 'IGOR' is announced."
    },
    {
        "id": "hyphen_band",
        "category": "Hyphen in Band Name",
        "title": "blink-182 - Enema of the State",
        "expected_release": True,
        "desc": "Artist is 'blink-182' without breaking on hyphen."
    },
    {
        "id": "numbered_zoetrope",
        "category": "Zoetrope & Retailer Exclusive",
        "title": "The 1975 - Live at Gorilla (Clear 2LP, Hand-Numbered /500) [Blood Records Zoetrope]",
        "expected_release": True,
        "desc": "Retailer 'Blood Records', variant captures clear 2LP and numbered /500 zoetrope."
    },
    {
        "id": "generic_restock_ignore",
        "category": "Generic Restock (No Albums Named) -> IGNORED",
        "title": "VMP Restocks: Fiona Apple, Outkast, The Mars Volta, DMX",
        "expected_release": False,
        "desc": "Should be ignored (is_vinyl_release=False, 0 releases) because no specific album titles are announced."
    },
    {
        "id": "storefront_sale_ignore",
        "category": "Storefront Sale (No Albums Named) -> IGNORED",
        "title": "Target Buy 2 Get 1 Free Sale on All Vinyl Records",
        "expected_release": False,
        "desc": "Store discount announcement without specific releases."
    },
    {
        "id": "discussion_ignore",
        "category": "Community Discussion -> IGNORED",
        "title": "[Discussion] What's your favorite pressing of In Rainbows?",
        "expected_release": False,
        "desc": "Community discussion filter."
    },
    {
        "id": "iso_ignore",
        "category": "In Search Of ([ISO]) -> IGNORED",
        "title": "[ISO] Looking for Blonde by Frank Ocean (Official Black Friday edition)",
        "expected_release": False,
        "desc": "Secondary marketplace request filter."
    },
]


def run_benchmark():
    agent = create_extraction_agent()
    print("=================================================================")
    print("RUNNING REFINED DEEPSEEK PYDANTIC-AI BENCHMARK")
    print("=================================================================\n")

    results_data = []

    for idx, case in enumerate(BENCHMARK_CASES, start=1):
        raw_title = case["title"]
        cleaned_title = pre_clean_title(raw_title)
        prompt = f"Extract release information from Reddit post title:\n\nOriginal: {raw_title}\nPre-cleaned: {cleaned_title}"

        print(f"[{idx}/{len(BENCHMARK_CASES)}] Testing: {case['category']}")
        print(f"      Title: \"{raw_title}\"")

        try:
            run_result = agent.run_sync(prompt)
            data: ExtractionResult = run_result.output
            
            dumped = data.model_dump()
            case_record = {
                "case": case,
                "result": dumped,
                "error": None,
            }
            results_data.append(case_record)

            print(f"      -> is_vinyl_release: {data.is_vinyl_release}")
            print(f"      -> releases found: {len(data.releases)}")
            for r in data.releases:
                print(f"         * Artists: {r.artists} (combined: '{r.combined_artist}') | Album: '{r.album_title}' | Variant: '{r.variant}' | Retailer: '{r.retailer}'")
            if data.reasoning:
                print(f"      -> Reasoning: {data.reasoning}")
        except Exception as e:
            print(f"      -> ERROR: {e}")
            results_data.append({
                "case": case,
                "result": None,
                "error": str(e)
            })
        print()

    output_path = os.path.join(os.path.dirname(__file__), "prototype_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Benchmark results written to: {output_path}")
    return results_data


if __name__ == "__main__":
    run_benchmark()
