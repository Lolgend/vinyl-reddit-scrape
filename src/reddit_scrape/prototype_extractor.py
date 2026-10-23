"""
PROTOTYPE: DeepSeek PydanticAI Extraction Schema & Prompt (Issue #5)
-------------------------------------------------------------------
This is throwaway prototype code designed to answer:
"Does a minimal PydanticAI prototype using deepseek-chat accurately parse complex
vinyl release post titles (including multi-artist releases, pre-orders, and complex
color variants) into structured ExtractedRelease objects without hallucinating
artists or album titles?"
"""

import json
import os
import re
from typing import Literal, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()


# --- Domain Extraction Schemas (from docs/research/vinyl-releases-post-patterns.md) ---

class ExtractedRelease(BaseModel):
    artist: str = Field(
        description="Name of musical artist, band, composer, or 'Various Artists'"
    )
    album_title: str = Field(
        description="Clean title of the album, EP, single, soundtrack, or box set without variant/date noise"
    )
    format_variant: Optional[str] = Field(
        None, description="Raw variant text, e.g. 'Pink Swirl 2LP', '180g Audiophile', 'Zoetrope Picture Disc'"
    )
    color_variant: Optional[str] = Field(
        None, description="Primary colorway description, e.g. 'Pink Swirl', 'Black Ice', 'Emerald', 'Clear'"
    )
    limitation_count: Optional[int] = Field(
        None, description="Total limited pressing count as an integer if specified (e.g. 500 for /500 or LTD 2000)"
    )
    retailer: Optional[str] = Field(
        None, description="Exclusive seller, record store, or label (e.g., 'Blood Records', 'Target', 'Rough Trade', 'VMP')"
    )
    is_soundtrack: bool = Field(
        False, description="Whether this release represents a video game, film, or television score/OST"
    )
    release_type: Literal[
        'new_release', 'pre_order', 'restock', 'repress', 'bootleg', 'unknown'
    ] = Field(
        'new_release', description="The release event classification"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Extraction confidence score from 0.0 to 1.0"
    )


class ExtractionResult(BaseModel):
    is_vinyl_release: bool = Field(
        description="Whether this submission represents an actual vinyl release announcement or drop. False if discussion, question, ISO, gear, etc."
    )
    releases: list[ExtractedRelease] = Field(
        default_factory=list,
        description="List of distinct releases announced in the submission title. Multiple if a batch/restock announcement."
    )
    reasoning: Optional[str] = Field(
        None, description="Brief explanation of extraction decisions and edge-case handling"
    )


# --- Pre-cleaning Logic ---

def pre_clean_title(raw_title: str) -> str:
    """Normalize unicode dashes, whitespace, and clean leading noise."""
    cleaned = raw_title.strip()
    # Normalize unicode en/em-dashes and figure dashes to standard hyphen
    cleaned = re.sub(r"[\u2010-\u2015\u2212]", "-", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


# --- Agent Factory ---

SYSTEM_PROMPT = """You are an expert vinyl record discography classifier specialized in extracting structured data from r/vinylreleases submission titles.

Community title conventions roughly follow:
  [Prefix Tags] [Date/Region] Artist - Title (Variant / Pressing Info) [Retailer / Price / Limitation]

Rules:
1. `is_vinyl_release`:
   - Set to False if the post is a community Discussion, Question, In Search Of ([ISO]), gear/turntable advice, or meta thread.
   - Set to True for announcements of new releases, pre-orders, represses, restocks, or bootlegs.
2. `releases`:
   - Extract all distinct releases. For batch restocks (e.g., "VMP Restocks: Artist A, Artist B"), extract an ExtractedRelease for each artist/title.
   - `artist`: Clean artist name. For split releases, join with '/' (e.g. "Converge / Dropdead"). For soundtracks with no composer named, use the game/franchise name or 'Various Artists' and set `is_soundtrack=True`.
   - `album_title`: Clean title without variant tags, dates, or prices. Do NOT hallucinate an album title if only an artist is given.
   - `color_variant`: Extract the specific colorway (e.g. "Opaque Emerald", "Black Ice", "Pink").
   - `limitation_count`: Parse limited edition quantities into a plain integer (e.g., "/500" -> 500, "Limited to 2,000" -> 2000).
   - `retailer`: Exclusive seller or store (e.g., "Rough Trade", "Target", "Blood Records", "VMP").
   - `release_type`: 'new_release', 'pre_order', 'restock', 'repress', 'bootleg', or 'unknown'.
   - `confidence`: 0.0 to 1.0 based on clarity of title.
3. NEVER hallucinate artists or album titles not present or clearly referenced in the post title.
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


# --- Benchmark Suite ---

BENCHMARK_CASES = [
    {
        "id": "canonical",
        "category": "Canonical Subreddit Rule #3",
        "title": "Fontaines D.C. - Romance (Opaque Emerald LP) [$32.99]",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "Fontaines D.C.",
            "album_title": "Romance",
            "color_variant": "Opaque Emerald",
            "retailer": None,
        }
    },
    {
        "id": "preorder_limitation",
        "category": "Pre-Order & Numbered Limitation",
        "title": "[Pre-Order] [10/24] Charli XCX - Brat (Translucent Black Ice Vinyl /2000) [Rough Trade]",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "Charli XCX",
            "album_title": "Brat",
            "limitation_count": 2000,
            "retailer": "Rough Trade",
            "release_type": "pre_order",
        }
    },
    {
        "id": "retailer_exclusive",
        "category": "Retailer Exclusive & Colorway",
        "title": "Chappell Roan - The Rise and Fall of a Midwest Princess (Pink Target Exclusive LP)",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "Chappell Roan",
            "album_title": "The Rise and Fall of a Midwest Princess",
            "color_variant": "Pink",
            "retailer": "Target",
        }
    },
    {
        "id": "zoetrope_numbered",
        "category": "Hand-Numbered Zoetrope Limitation",
        "title": "The 1975 - Live at Gorilla (Clear 2LP, Hand-Numbered /500) [Blood Records Zoetrope]",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "The 1975",
            "album_title": "Live at Gorilla",
            "limitation_count": 500,
            "retailer": "Blood Records",
        }
    },
    {
        "id": "unicode_dash",
        "category": "Unicode En-dash & Audiophile",
        "title": "Miles Davis – Kind of Blue (Mobile Fidelity 180g 45RPM 2LP Box)",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "Miles Davis",
            "album_title": "Kind of Blue",
            "format_variant": "Mobile Fidelity 180g 45RPM 2LP Box",
        }
    },
    {
        "id": "comma_artist_restock",
        "category": "Internal Comma in Artist & Restock Tag",
        "title": "RESTOCK: Tyler, The Creator - IGOR (Mint Green Special Edition LP)",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "Tyler, The Creator",
            "album_title": "IGOR",
            "release_type": "restock",
            "color_variant": "Mint Green",
        }
    },
    {
        "id": "hyphenated_band",
        "category": "Hyphen in Band Name",
        "title": "blink-182 - Enema of the State",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "blink-182",
            "album_title": "Enema of the State",
        }
    },
    {
        "id": "soundtrack_no_hyphen",
        "category": "Video Game OST without Artist Delimiter",
        "title": "The Elder Scrolls IV: Oblivion (Game Soundtrack) (Oblivion Gate Orange 2LP)",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "is_soundtrack": True,
            "color_variant": "Oblivion Gate Orange",
        }
    },
    {
        "id": "composer_soundtrack",
        "category": "Film Score with Composer",
        "title": "Hans Zimmer - Dune: Part Two (Original Motion Picture Soundtrack)",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "Hans Zimmer",
            "album_title": "Dune: Part Two",
            "is_soundtrack": True,
        }
    },
    {
        "id": "split_release",
        "category": "Split Release Multi-Artist",
        "title": "Converge / Dropdead - Split (7\" /500)",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "Converge / Dropdead",
            "limitation_count": 500,
        }
    },
    {
        "id": "self_titled",
        "category": "Self-Titled Album",
        "title": "Paramore - Paramore (Indie Exclusive Tangerine Vinyl)",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "Paramore",
            "album_title": "Paramore",
            "color_variant": "Tangerine",
        }
    },
    {
        "id": "bootleg",
        "category": "Bootleg Tagging",
        "title": "[BOOTLEG] Frank Ocean - Nostalgia, Ultra (Orange Vinyl)",
        "expected": {
            "is_vinyl_release": True,
            "count": 1,
            "artist": "Frank Ocean",
            "album_title": "Nostalgia, Ultra",
            "release_type": "bootleg",
            "color_variant": "Orange",
        }
    },
    {
        "id": "multi_item_dump",
        "category": "Batch Multi-Item Restock Dump",
        "title": "VMP Restocks: Fiona Apple, Outkast, The Mars Volta, DMX",
        "expected": {
            "is_vinyl_release": True,
            "retailer": "VMP",
            "release_type": "restock",
            "min_releases": 4,
        }
    },
    {
        "id": "discussion_filter",
        "category": "Discussion Filter (Non-Release)",
        "title": "[Discussion] What's your favorite pressing of In Rainbows?",
        "expected": {
            "is_vinyl_release": False,
            "count": 0,
        }
    },
    {
        "id": "iso_filter",
        "category": "In Search Of (ISO) Filter (Non-Release)",
        "title": "[ISO] Looking for Blonde by Frank Ocean (Official Black Friday edition)",
        "expected": {
            "is_vinyl_release": False,
            "count": 0,
        }
    },
]


def run_benchmark():
    agent = create_extraction_agent()
    print("=================================================================")
    print("RUNNING DEEPSEEK PYDANTIC-AI EXTRACTION BENCHMARK")
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
            
            # Serialize for reporting
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
                print(f"         * Artist: '{r.artist}' | Album: '{r.album_title}' | Variant: '{r.color_variant}' | Limit: {r.limitation_count} | Retailer: '{r.retailer}' | Type: {r.release_type} (conf: {r.confidence})")
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

    # Save benchmark results to json for analysis and HTML demo embedding
    output_path = os.path.join(os.path.dirname(__file__), "prototype_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    print(f"Benchmark results written to: {output_path}")
    return results_data


if __name__ == "__main__":
    run_benchmark()
