"""
PROTOTYPE: DeepSeek PydanticAI Extraction Schema & Prompt (Issue #5 - Minimal)
-----------------------------------------------------------------------------
Streamlined to store strictly processed release entities:
- Removed reasoning, clean_title, confidence, is_soundtrack, limitation_count, release_type.
- If a post is not a specific release announcement (e.g. discussion, ISO, generic restock without album names), returns empty releases: [].
- Fields per release: artists (list[str]), combined_artist (Optional[str]), album_title (str), variant (Optional[str]), retailer (Optional[str]).
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


# --- Minimal Domain Extraction Schemas ---

class ExtractedRelease(BaseModel):
    artists: list[str] = Field(
        description="Individual artist or band names contributing to the release (e.g. ['Converge', 'Dropdead'] or ['Tyler, The Creator'])"
    )
    combined_artist: Optional[str] = Field(
        None,
        description="Combined billing string for multi-artist/collaborative releases (e.g. 'Converge / Dropdead', 'Kanye West & Ty Dolla $ign')"
    )
    album_title: str = Field(
        description="Album, EP, or single title"
    )
    variant: Optional[str] = Field(
        None,
        description="Physical pressing variation (e.g. 'Opaque Emerald LP', '180g 45RPM 2LP Box', 'Pink /500')"
    )
    retailer: Optional[str] = Field(
        None,
        description="Exclusive retailer, store, or record label if mentioned (e.g. 'Rough Trade', 'Blood Records', 'Target')"
    )


class ExtractionResult(BaseModel):
    releases: list[ExtractedRelease] = Field(
        default_factory=list,
        description="List of distinct releases announced with specific album titles. Empty if non-release or generic store restock."
    )


# --- Pre-cleaning Logic ---

def pre_clean_title(raw_title: str) -> str:
    """Normalize unicode dashes and whitespace."""
    cleaned = raw_title.strip()
    cleaned = re.sub(r"[\u2010-\u2015\u2212]", "-", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


# --- Agent Factory ---

SYSTEM_PROMPT = """You are an expert vinyl record discography classifier specialized in extracting structured data from r/vinylreleases submission titles.

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
        "category": "Canonical Post",
        "title": "Fontaines D.C. - Romance (Opaque Emerald LP) [$32.99]",
        "desc": "Standard single artist release with colorway variant."
    },
    {
        "id": "preorder_variant",
        "category": "Pre-Order & Retailer Exclusive",
        "title": "[Pre-Order] [10/24] Charli XCX - Brat (Translucent Black Ice Vinyl /2000) [Rough Trade]",
        "desc": "Variant captures translucent black ice and /2000 limitation text, retailer Rough Trade."
    },
    {
        "id": "audiophile_weight",
        "category": "Audiophile Weight Variant (180g / 45RPM)",
        "title": "Miles Davis – Kind of Blue (Mobile Fidelity 180g 45RPM 2LP Box)",
        "desc": "Non-color physical variant (180g, 45RPM, box)."
    },
    {
        "id": "multi_artist_split",
        "category": "Multi-Artist Split Release",
        "title": "Converge / Dropdead - Split (7\" /500)",
        "desc": "Split release: individual artists ['Converge', 'Dropdead'] and combined 'Converge / Dropdead'."
    },
    {
        "id": "multi_artist_collab",
        "category": "Multi-Artist Collaboration",
        "title": "Kanye West & Ty Dolla $ign - Vultures 1 (Black 2LP)",
        "desc": "Collaboration: individual artists ['Kanye West', 'Ty Dolla $ign'] and combined relation."
    },
    {
        "id": "comma_artist_restock",
        "category": "Internal Comma in Artist with Specific Album",
        "title": "RESTOCK: Tyler, The Creator - IGOR (Mint Green Special Edition LP)",
        "desc": "Specific album 'IGOR' restock with pressing variant."
    },
    {
        "id": "hyphen_band",
        "category": "Hyphen in Band Name",
        "title": "blink-182 - Enema of the State",
        "desc": "Band name 'blink-182' intact."
    },
    {
        "id": "numbered_zoetrope",
        "category": "Zoetrope & Retailer Exclusive",
        "title": "The 1975 - Live at Gorilla (Clear 2LP, Hand-Numbered /500) [Blood Records Zoetrope]",
        "desc": "Retailer 'Blood Records', variant captures clear 2LP, hand-numbered /500, zoetrope."
    },
    {
        "id": "generic_restock_ignore",
        "category": "Generic Restock (No Albums Named) -> Ignored",
        "title": "VMP Restocks: Fiona Apple, Outkast, The Mars Volta, DMX",
        "desc": "No specific album titles -> returns empty releases: []."
    },
    {
        "id": "storefront_sale_ignore",
        "category": "Storefront Sale (No Albums Named) -> Ignored",
        "title": "Target Buy 2 Get 1 Free Sale on All Vinyl Records",
        "desc": "Store discount announcement -> returns empty releases: []."
    },
    {
        "id": "discussion_ignore",
        "category": "Community Discussion -> Ignored",
        "title": "[Discussion] What's your favorite pressing of In Rainbows?",
        "desc": "Community discussion -> returns empty releases: []."
    },
    {
        "id": "iso_ignore",
        "category": "In Search Of ([ISO]) -> Ignored",
        "title": "[ISO] Looking for Blonde by Frank Ocean (Official Black Friday edition)",
        "desc": "Secondary marketplace request -> returns empty releases: []."
    },
]


def run_benchmark():
    agent = create_extraction_agent()
    print("=================================================================")
    print("RUNNING MINIMAL DEEPSEEK PYDANTIC-AI BENCHMARK")
    print("=================================================================\n")

    results_data = []

    for idx, case in enumerate(BENCHMARK_CASES, start=1):
        raw_title = case["title"]
        cleaned_title = pre_clean_title(raw_title)
        prompt = f"Extract release entities from Reddit post title:\n\n{cleaned_title}"

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

            print(f"      -> releases found: {len(data.releases)}")
            for r in data.releases:
                print(f"         * Artists: {r.artists} (combined: '{r.combined_artist}') | Album: '{r.album_title}' | Variant: '{r.variant}' | Retailer: '{r.retailer}'")
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
