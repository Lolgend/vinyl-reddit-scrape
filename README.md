# Reddit Vinyl Releases Scraper

Continuous dual-loop Reddit scraper, structured entity extraction pipeline, and two-pass **3-Tier RAG Catalog Enrichment** engine for `r/vinylreleases`. Ingests community **Submissions**, tracks engagement velocity and flair mutations, aligns against a normalized 3-tier relational catalog (**Artist** $\leftrightarrow$ **Album** $\leftrightarrow$ **Release**) in PostgreSQL (`pgvector` + local CPU `fastembed`), and asynchronously enriches musical works against the Discogs Database API and DeepSeek Native Web Search.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Reddit["Reddit Community Source"]
        RN["r/vinylreleases/new (Discovery)"]
        RH["r/vinylreleases/hot (Tracking)"]
    end

    subgraph Poller["PollerService"]
        PNew["Discovery Poller Loop"]
        PHot["Tracking Poller Loop"]
        ExtWorker["Pass 1: ExtractionWorker Thread"]
        EnrWorker["Pass 2: DiscogsEnrichmentWorker Thread"]
    end

    subgraph Pass1["Pass 1: Extraction & Line 1 Local DB Vector RAG"]
        Batcher["Extraction Batch Partitioning (size: llm_batch_size)"]
        Line1RAG["Line 1 Vector Lookup (FastEmbed + pgvector)\nTop-5 Albums + Artists + Hydrated Sibling Releases"]
        LLMExtract["DeepSeek Extraction Agent\n(Clean Work Title vs. Edition Variant Separation)"]
        Fallback["Single-Submission Fallback"]
        PersistP1["Persist Artist / Album (pending + provisional 384-d vector) / Release"]
    end

    subgraph Pass2["Pass 2: Detached 4-Phase Catalog Enrichment"]
        Queue["Priority Queue Selector (idx_albums_enrichment_queue)\nP0: pending | P1: due failed (exp. backoff) | P2: due partial_matched (7d/60d)"]
        Line2Cascade["Line 2: 5-Tier Discogs Search Cascade\n(1.15s rate limit + header backpressure + 429/5xx retries)"]
        Arbiter["LLM RAG Candidate Arbiter + Master Promotion\n(<=5 candidates + FastEmbed cosine similarity)"]
        Hydrate["1 Follow-Up Detail Hydration\n(GET /masters/{id} or GET /releases/{id})\n+ LLM Homonym (n) Disambiguation"]
        Line3Web["Line 3: DeepSeek Native Web Search Fallback\n(POST /v1/tools/search, count=10, need_content=true)"]
        WriteTx1["Phase 3 (Write Tx 1): Canonical Artist/Album Update,\nIn-Place Embedding Overwrite & Collision Re-Parenting"]
        Phase4Sibling["Phase 4 (Write Tx 2): LLM Sibling Release RAG Reconciliation\n(Merge duplicate variant/retailer rows when >=2 child Releases)"]
    end

    subgraph Database["PostgreSQL + pgvector (3-Tier Schema)"]
        SubTable["submissions & submission_snapshots"]
        CatTable["artists, albums, album_artists,\nreleases & submission_releases\n(384-d HNSW + partial queue indexes)"]
    end

    RN -->|Poll new| PNew
    RH -->|Poll hot| PHot
    PNew -->|Upsert submissions & log discoveries| SubTable
    PHot -->|Update engagement & mutate flairs| SubTable
    PNew -.->|Signal _extraction_trigger| ExtWorker
    ExtWorker --> Batcher --> Line1RAG
    Line1RAG <-->|Cosine >= 0.50| CatTable
    Line1RAG -->|Existing Catalog Reference| LLMExtract
    LLMExtract -->|Success| PersistP1
    LLMExtract -.->|Error or Omission| Fallback --> PersistP1
    PersistP1 -->|Upsert 3-tier rows| CatTable
    PersistP1 -.->|Signal _enrichment_trigger| EnrWorker
    EnrWorker --> Queue
    Queue <-->|Phase 1 Short Read Tx| CatTable
    Queue --> Line2Cascade --> Arbiter
    Arbiter -->|Work Match (Tiers 1-4)| Hydrate --> WriteTx1
    Arbiter -->|Unmatched Work (partial_matched / not_found)| Line3Web --> WriteTx1
    WriteTx1 -->|Commit canonical state & re-parent collisions| CatTable
    WriteTx1 --> Phase4Sibling -->|Merge duplicate sibling Releases| CatTable
```

### Core Pipeline Flows

1. **Dual-Loop Ingestion**:
   - `/new` discovery polls ingest newly posted community **Submissions** and initial **Submission Snapshots**.
   - `/hot` tracking polls monitor active threads, recording engagement velocity (`score`, `num_comments`, `upvote_ratio`) and flair mutations (e.g., `SOLD OUT`, `RESTOCK`).
2. **Normalized 3-Tier Relational Catalog (`Artist` $\leftrightarrow$ `Album` $\leftrightarrow$ `Release`)**:
   - **`Artist` (`artists`)**: Canonical musical creator with optional `discogs_artist_id` and 384-d HNSW vector index (`idx_artists_embedding`).
   - **`Album` (`albums`)**: Canonical musical work (`combined_artist`, `title`, deterministic SHA-256 `fingerprint` over `"{norm_combined_artist}|{norm_title}"`, `discogs_master_id`, `discogs_release_id`, `discogs_status` **Enrichment State**, retry metadata, and 384-d HNSW vector index `idx_albums_embedding`). Linked to contributing creators via `album_artists`.
   - **`Release` (`releases`)**: Physical vinyl pressing or commercial edition child of an **Album** (`album_id` FK, `variant`, `retailer_name`, and deterministic SHA-256 `fingerprint` over `"{album_id}|{norm_variant}|{norm_retailer}"`). Linked to announcing Reddit posts via `submission_releases`.
3. **Pass 1 — Line 1 Local DB Vector RAG, Clean Work Title Separation & Provisional Embeddings**:
   - Before LLM extraction, **Submission** titles are embedded locally on CPU via `fastembed` (`BAAI/bge-small-en-v1.5`, 384-d) and matched against `Album.embedding` and `Artist.embedding` in PostgreSQL ($\text{cosine similarity} \ge 0.50$).
   - Matched **Albums** are hydrated with their contributing **Artists** and all existing sibling **Release** `(variant, retailer_name)` pairs into the prompt's `Existing Catalog Reference` block for **Canonical Alignment**.
   - The Pass 1 extraction prompts isolate only the core canonical musical work title into `album_title` while routing anniversary, remaster, reissue, deluxe, and expanded edition descriptors into `variant` alongside colorway, weight, LP count, and run size—preventing edition token noise from breaking external catalog searches.
   - Newly persisted **Albums** receive an immediate **provisional** 384-d embedding (`discogs_status = 'pending'`) so subsequent **Submissions** for the same drop arriving minutes later align immediately in Line 1 Vector RAG before Pass 2 finishes. Re-ingesting or adding a new **Release** to an already enriched **Album** preserves its canonical `Album.embedding` and `discogs_status`.
4. **Pass 2 — Decoupled `DiscogsEnrichmentWorker` & 3-Tier RAG Resolution Cascade**:
   - **Priority Queue State Machine**: Selects the next eligible **Album** via partial index `idx_albums_enrichment_queue` in strict priority order:
     - **Priority 0 (`pending`)**: Newly discovered **Albums**.
     - **Priority 1 (`failed`)**: Transient failures with $\text{attempts} < 5$ and exponential backoff elapsed ($5\text{m} \times 2^{\text{attempts}-1}$: 5m, 10m, 20m, 40m; dead-lettered at 5 attempts).
     - **Priority 2 (`partial_matched`)**: Pre-order **Albums** by known Discogs **Artists** re-checked every 7 days for up to 60 days after `created_at`.
   - **4-Phase Detached Transaction Protocol**:
     - **Phase 1 (Short Read Tx)**: Reads the eligible **Album** and its linked **Artists**, then closes the database session.
     - **Phase 2 (Detached Network & CPU I/O — Zero Open DB Session)**:
       - **Line 2 (5-Tier Discogs Search Cascade)**: Queries `GET /database/search` (`per_page=5`) across up to 5 tiers (`Tier 1`: structured `type=master` $\rightarrow$ `Tier 2`: structured `type=release&format=Vinyl` $\rightarrow$ `Tier 3`: keyword `type=master&q=` $\rightarrow$ `Tier 4`: keyword `type=release&format=Vinyl&q=` $\rightarrow$ `Tier 5`: per-constituent `type=artist&q=` for unlinked **Artists**), paced by a thread-safe 1.15s per-request rate limiter with adaptive header backpressure (`X-Discogs-Ratelimit-Remaining <= 5`) and `HTTP 429`/`5xx` `Retry-After` retries.
       - **LLM RAG Candidate Arbiter & Detail Hydration**: Passes deduplicated candidates with local `fastembed` cosine similarity scores to the DeepSeek arbiter (with automatic `master_id` promotion from `type="release"` hits), then executes **1 follow-up detail call** (`GET /masters/{id}` or `GET /releases/{id}`) and uses LLM reconciliation to strip Discogs numeric homonym suffixes (`Nirvana (2)` $\rightarrow$ `Nirvana`) while preserving legitimate parenthetical band names (`Sunn O)))`) and reconstructing `combined_artist` via `artists[].join`.
       - **Line 3 (DeepSeek Native Web Search Fallback)**: When Tiers 1–4 do not match the **Album** (`partial_matched` if $\ge 1$ **Artist** matched on Discogs, else `not_found`), queries `POST {deepseek_base_url}/v1/tools/search` (`count=10`, `need_content=True`) and reconciles official artist/album capitalization and titles via LLM (falling back gracefully to Pass 1 names if web search fails).
     - **Phase 3 (Short Write Tx 1 — Canonical Commit & Collision Re-Parenting)**: Overwrites `Album.embedding` and `Artist.embedding` in-place with canonical 384-d vectors, updates `discogs_status`, and automatically re-parents `album_artists`, child `releases`, and `submission_releases` if canonical updates collide with an already-existing **Artist** or **Album**.
     - **Phase 4 (Detached Sibling Release RAG Reconciliation + Short Write Tx 2)**: When the surviving **Album** has $\ge 2$ child **Releases**, runs an LLM sibling reconciliation pass outside any DB session and merges duplicate `(variant, retailer_name)` rows in a short second write transaction.

---

## Installation & Environment Setup

The project uses [`uv`](https://github.com/astral-sh/uv) as the package manager and runner.

```powershell
# Clone the repository
git clone https://github.com/Lolgend/vinyl-reddit-scrape.git
cd vinyl-reddit-scrape

# Create virtual environment and sync dependencies
uv sync
```

### Environment Variables

Configure a `.env` file at the repository root:

| Variable | Required | Default | Description |
|---|---|---|---|
| `REDDIT_CLIENT_ID` | **Yes** | - | Reddit Script Application client ID |
| `REDDIT_CLIENT_SECRET` | **Yes** | - | Reddit Script Application client secret |
| `REDDIT_USER_AGENT` | No | `python:vinyl-releases-scraper:v0.1.0` | User agent header sent to Reddit API |
| `DATABASE_URL` | **Yes** | - | PostgreSQL connection URL (e.g. `postgresql://user:pass@localhost:5432/vinyl`) |
| `DEEPSEEK_API_KEY` | **Yes** | - | API key for DeepSeek LLM & Search API (alias: `DEEPSEEK_KEY`) |
| `DEEPSEEK_BASE_URL` | No | `https://api.deepseek.com` | Base URL for DeepSeek LLM & `/v1/tools/search` endpoints |
| `DEEPSEEK_MODEL` | No | `deepseek-chat` | Model name for extraction and enrichment arbitration |
| `DISCOGS_TOKEN` | **Conditional*** | - | Discogs Personal Access Token (preferred if both token and key/secret are set) |
| `DISCOGS_CONSUMER_KEY` | **Conditional*** | - | Discogs application Consumer Key (required with `DISCOGS_CONSUMER_SECRET` if `DISCOGS_TOKEN` is unset) |
| `DISCOGS_CONSUMER_SECRET` | **Conditional*** | - | Discogs application Consumer Secret (required with `DISCOGS_CONSUMER_KEY` if `DISCOGS_TOKEN` is unset) |
| `DISCOGS_USER_AGENT` | No | `VinylReleasesScraper/0.1.0` | User-Agent header sent to Discogs API (`https://api.discogs.com`) |
| `ENRICHMENT_INTERVAL_SECONDS` | No | `60` | Background idle sweep interval for `DiscogsEnrichmentWorker` (positive integer, seconds) |
| `LLM_THINKING_LEVEL` | No | Model-aware (`none` for chat, `low` for reasoning) | Reasoning effort (`none`, `minimal`, `low`, `medium`, `high`, `max`) |
| `LLM_BATCH_SIZE` | No | `5` | Submissions batched per Pass 1 LLM extraction call (positive integer) |
| `POLL_INTERVAL_SECONDS` | No | `300` | Discovery poll interval for `/new` (seconds) |
| `HOT_POLL_INTERVAL_SECONDS` | No | `900` | Metric tracking poll interval for `/hot` (seconds) |
| `SNAPSHOT_RETENTION_DAYS` | No | `30` | Default retention threshold for snapshot pruning |

*\*Either `DISCOGS_TOKEN` or both `DISCOGS_CONSUMER_KEY` and `DISCOGS_CONSUMER_SECRET` must be provided.*

### Database Setup & Schema Initialization

PostgreSQL tables (`submissions`, `submission_snapshots`, `artists`, `albums`, `album_artists`, `releases`, `submission_releases`), the `pgvector` extension (`vector`), HNSW cosine distance indexes (`idx_albums_embedding`, `idx_artists_embedding`), and the partial enrichment queue index (`idx_albums_enrichment_queue`) are managed declaratively via SQLAlchemy metadata:

- **Prerequisites**: PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension installed.
- **Automatic Initialization**: When running the poller service (`reddit-scrape run`), `reddit_scrape.db.init_db()` is called automatically on startup to enable `CREATE EXTENSION IF NOT EXISTS vector`, drop any incompatible legacy 2-tier catalog tables if migrating from an older schema, and ensure all 3-tier relational tables and indexes exist.
- **Manual Initialization / Clean Reset**:
  ```powershell
  # Idempotent schema initialization
  uv run python -c "from reddit_scrape.db import init_db; init_db()"

  # Clean drop and re-creation of all tables and vector indexes
  uv run python -c "from reddit_scrape.db import reset_db; reset_db()"
  ```

---

## CLI Reference

The CLI entrypoint is `reddit-scrape` (or `uv run python -m reddit_scrape.cli`).

### `run` Subcommand

Starts the continuous ingestion, Pass 1 extraction, and Pass 2 Discogs/Web catalog enrichment service (or runs a single synchronous end-to-end cycle with `--once`).

```powershell
# Start continuous service (spawns ExtractionWorker and DiscogsEnrichmentWorker daemon threads)
uv run python -m reddit_scrape.cli run -v

# Run a single discovery, extraction, and enrichment cycle synchronously and exit
# Logs: Single cycle complete: new_discovered=<n>, new_updated=<n>, hot_updated=<n>, extracted=<n>, enriched=<n>
uv run python -m reddit_scrape.cli run --once -v

# Custom batch size and reasoning effort
uv run python -m reddit_scrape.cli run --llm-batch-size 8 --thinking-level medium --log-level INFO
```

#### Available Flags for `run`:

- `--once`: Execute a single poll, Pass 1 extraction, and Pass 2 catalog enrichment cycle synchronously, print summary metrics (`new_discovered`, `new_updated`, `hot_updated`, `extracted`, `enriched`), and exit.
- `-v`, `--verbose`: Enable verbose stream logging (configures root logger to `INFO` level with standard timestamped format).
- `--log-level`: Set explicit logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `--llm-batch-size`: Number of submissions batched per LLM extraction prompt (default: `LLM_BATCH_SIZE` or `5`).
- `--thinking-level`: LLM reasoning effort level (`none`, `minimal`, `low`, `medium`, `high`, `max`; default: `LLM_THINKING_LEVEL` or model-aware default).
- `--new-interval`: Polling interval in seconds for `/new` (default: `POLL_INTERVAL_SECONDS` or `300`).
- `--hot-interval`: Polling interval in seconds for `/hot` (default: `HOT_POLL_INTERVAL_SECONDS` or `900`).
- `--new-limit`: Number of submissions to fetch per `/new` poll (default: `50`).
- `--hot-limit`: Number of submissions to fetch per `/hot` poll (default: `50`).

### `purge-snapshots` Subcommand

Prunes historical engagement snapshots older than the specified retention window while preserving submission and catalog records.

```powershell
# Purge snapshots older than 30 days (default)
uv run python -m reddit_scrape.cli purge-snapshots

# Purge snapshots older than 14 days with chunked batching
uv run python -m reddit_scrape.cli purge-snapshots --days 14 --batch-size 10000
```

---

## Documentation & Architecture References

- **Domain Glossary**: [`GLOSSARY.md`](GLOSSARY.md)
- **Technical Specifications**:
  - [`docs/spec/ingestion-and-extraction-pipeline.md`](docs/spec/ingestion-and-extraction-pipeline.md)
  - [`docs/spec/discogs-enrichment-and-artist-catalog.md`](docs/spec/discogs-enrichment-and-artist-catalog.md)
- **Architecture Decision Records (`docs/adr/`)**:
  - [ADR-0004: Normalized 3-Tier Artist–Album–Release Schema & Deterministic Fingerprinting](docs/adr/0004-three-tier-artist-album-release-schema.md)
  - [ADR-0006: Array Batch Extraction with Automated Single-Submission Fallback](docs/adr/0006-array-batch-extraction-with-fallback.md)
  - [ADR-0007: Pre-Retrieval Vector Context for Canonical Entity Alignment](docs/adr/0007-vector-canonical-alignment.md)
  - [ADR-0008: 3-Tier RAG Resolution Cascade for Catalog Enrichment (Local Vector DB + Discogs API + DeepSeek Web Search)](docs/adr/0008-three-tier-rag-catalog-enrichment.md)

---

## Development & Testing

```powershell
# Run the test suite
uv run pytest

# Type check
uv run mypy src tests

# Code formatting and linting
uv run ruff check .
```
