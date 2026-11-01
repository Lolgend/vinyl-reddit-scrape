# Three-Tier RAG Cascade for Catalog Enrichment and Pre-Order Handling

Querying external music databases synchronously during Reddit ingestion risks rate-limit exhaustion (Discogs allows 60 req/min) and high latency spikes, while brand-new pre-order drops frequently lack Discogs records entirely. Pure algorithmic string-similarity heuristics also struggle with creative reissues, reordered split collaborations, and boutique retail editions.

We decided to resolve entities via a decoupled 3-tier RAG cascade:
1. **Line 1 (Local DB Vector RAG)**: Matches incoming Reddit submissions against local PostgreSQL `Album` embeddings (`fastembed` `BAAI/bge-small-en-v1.5` 384-d on `"{combined_artist} - {album_title}"`) to align existing works instantly with zero network latency.
2. **Line 2 (Discogs Search API RAG)**: A decoupled background worker queries Discogs at a paced 1.15s interval, injecting raw candidate summaries (`id`, `type`, `artist`, `title`) into an LLM RAG reconciliation prompt where the LLM standardizes canonical `Artist` and `Album` names from Discogs while isolating Reddit-specific `Variant` and `Retailer` details.
3. **Line 3 (LLM Web Search Fallback)**: Escalates ambiguous or unindexed pre-order drops to an LLM web search query (`"{extracted_artist} {extracted_album} album"`) returning structured canonical naming.

Unmatched albums in Pass 1 are saved with `discogs_status = 'pending'` with vector embeddings deferred (`embedding = NULL`). Only once verified and enriched in Pass 2 is an album's vector embedding generated for the first time, ensuring the Line 1 vector index remains an authoritative canonical reference. Pre-orders matching an `Artist` without an `Album` enter `discogs_status = 'partial_matched'` and are scheduled for periodic re-checks.
