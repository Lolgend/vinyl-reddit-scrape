# Reddit Vinyl Releases Scraper

Continuous dual-loop Reddit scraper and structured entity extraction pipeline for `r/vinylreleases`. Ingests community announcements, tracks engagement velocity and flair mutations, performs pre-retrieval semantic catalog lookup (`pgvector` + local CPU `fastembed`) for **Canonical Alignment**, extracts multi-artist discography releases using DeepSeek LLM via PydanticAI with array batching and automatic single-submission fallback, and persists normalized relational and vectorized vinyl entities in PostgreSQL.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Reddit["Reddit Community Source"]
        RN["r/vinylreleases/new (Discovery)"]
        RH["r/vinylreleases/hot (Tracking)"]
    end

    subgraph Service["PollerService"]
        PNew["Discovery Poller Loop"]
        PHot["Tracking Poller Loop"]
        BQueue["Unextracted Candidates Queue"]
        Worker["Decoupled Extraction Worker Thread"]
    end

    subgraph ExtractionEngine["Batch Extraction & Canonical Alignment"]
        Batcher["Batch Partitioning (size: llm_batch_size)"]
        RAG["Pre-Retrieval Vector Lookup (FastEmbed + pgvector)"]
        LLMBatch["DeepSeek LLM (Batch Array Prompt + Catalog Reference)"]
        Fallback["Single-Submission Fallback"]
    end

    subgraph Database["PostgreSQL + pgvector Storage"]
        SubTable["submissions & snapshots"]
        RelTable["releases, artists (384-d HNSW vectors) & join tables"]
    end

    RN -->|Poll new| PNew
    RH -->|Poll hot| PHot
    PNew -->|Upsert submissions & log discoveries| SubTable
    PHot -->|Update engagement & mutate flairs| SubTable
    PNew -.->|Signal unextracted ready| BQueue
    BQueue --> Batcher
    Batcher -->|Embed titles & query cosine >= 0.50| RAG
    RAG <-->|Top-5 Releases & Artists| RelTable
    RAG -->|Inject Existing Catalog Reference| LLMBatch
    LLMBatch -->|Success| Worker
    LLMBatch -.->|Error or Omission| Fallback
    Fallback --> Worker
    Worker -->|Embed & Persist Releases + Artists| RelTable
```

The pipeline operates with:
- **Dual-loop ingestion**: `/new` polls discover new release announcements, while `/hot` polls monitor engagement velocity and flair mutations (e.g. `SOLD OUT`).
- **Decoupled extraction worker**: Ingestion loops enqueue unextracted candidate submissions without blocking on LLM inference. A background worker processes candidates in bounded batches.
- **Pre-retrieval vector search & Canonical Alignment**: Prior to LLM extraction, submission titles are embedded locally on CPU via `fastembed` (`BAAI/bge-small-en-v1.5`, 384 dimensions) and matched against existing `releases` and `artists` in PostgreSQL (`pgvector` HNSW cosine distance index, similarity $\ge 0.50$). Top-5 candidate releases and artists are injected as an `"Existing Catalog Reference"` block so the LLM aligns reversed artist/album titles, reordered multi-artist billings, and intra-batch variations to canonical strings before deterministic SHA-256 fingerprinting.
- **Array batch extraction with automated fallback**: Bounded candidate batches (default: 5 submissions) are evaluated simultaneously in structured array prompts. If a batch call fails or omits submissions, the worker falls back to individual extraction per submission (preserving candidate context) to guarantee zero data loss.

---

## Installation & Environment Setup

The project uses [`uv`](https://github.com/astral-sh/uv) as the package manager and runner.

```powershell
# Clone the repository
git clone https://github.com/your-org/reddit-scrape.git
cd reddit-scrape

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
| `DEEPSEEK_API_KEY` | **Yes** | - | API key for DeepSeek LLM (alias: `DEEPSEEK_KEY`) |
| `DEEPSEEK_BASE_URL` | No | `https://api.deepseek.com` | Base URL for DeepSeek API endpoint |
| `DEEPSEEK_MODEL` | No | `deepseek-chat` | Model name for extraction (`deepseek-chat`) |
| `LLM_THINKING_LEVEL` | No | Model-aware (`none` for chat, `low` for reasoning) | Reasoning effort (`none`, `minimal`, `low`, `medium`, `high`, `max`) |
| `LLM_BATCH_SIZE` | No | `5` | Submissions batched per LLM extraction call (positive integer) |
| `POLL_INTERVAL_SECONDS` | No | `300` | Discovery poll interval for `/new` (seconds) |
| `HOT_POLL_INTERVAL_SECONDS` | No | `900` | Metric tracking poll interval for `/hot` (seconds) |
| `SNAPSHOT_RETENTION_DAYS` | No | `30` | Default retention threshold for snapshot pruning |

### Database Setup & Migrations

PostgreSQL tables, foreign key relationships, `pgvector` extension (`vector`), and HNSW cosine distance indexes (`ix_releases_embedding_hnsw`, `ix_artists_embedding_hnsw`) are managed declaratively via SQLAlchemy metadata:

- **Prerequisites**: PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension installed.
- **Automatic Initialization**: When running the poller service (`reddit-scrape run`), `reddit_scrape.db.init_db()` is called automatically on startup to enable `CREATE EXTENSION IF NOT EXISTS vector` and ensure all relational tables and HNSW vector indexes exist.
- **Manual Initialization / Clean Reset**: Tables can also be created or cleanly re-created directly via Python:
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

Starts the continuous ingestion and extraction poller service.

```powershell
# Start continuous service with verbose stream logging
uv run python -m reddit_scrape.cli run -v

# Run a single discovery and extraction cycle and exit
uv run python -m reddit_scrape.cli run --once -v

# Custom batch size and reasoning effort
uv run python -m reddit_scrape.cli run --llm-batch-size 8 --thinking-level medium --log-level INFO
```

#### Available Flags for `run`:

- `--once`: Execute a single poll and extraction cycle, print summary metrics, and exit.
- `-v`, `--verbose`: Enable verbose stream logging (configures root logger to `INFO` level with standard timestamped format).
- `--log-level`: Set explicit logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `--llm-batch-size`: Number of submissions batched per LLM extraction prompt (default: `LLM_BATCH_SIZE` or `5`).
- `--thinking-level`: LLM reasoning effort level (`none`, `minimal`, `low`, `medium`, `high`, `max`; default: `LLM_THINKING_LEVEL` or model-aware default).
- `--new-interval`: Polling interval in seconds for `/new` (default: `POLL_INTERVAL_SECONDS` or `300`).
- `--hot-interval`: Polling interval in seconds for `/hot` (default: `HOT_POLL_INTERVAL_SECONDS` or `900`).
- `--new-limit`: Number of submissions to fetch per `/new` poll (default: `50`).
- `--hot-limit`: Number of submissions to fetch per `/hot` poll (default: `50`).

### `purge-snapshots` Subcommand

Prunes historical engagement snapshots older than the specified retention window while preserving submission records and aggregated metrics.

```powershell
# Purge snapshots older than 30 days (default)
uv run python -m reddit_scrape.cli purge-snapshots

# Purge snapshots older than 14 days with chunked batching
uv run python -m reddit_scrape.cli purge-snapshots --days 14 --batch-size 10000
```

---

## Batch Extraction, Canonical Alignment & Resilience Engine

For high throughput during backlogs, `PollerService` partitions candidate submissions into bounded batches (`llm_batch_size`). Before each batch call, `ReleasePersistenceService.find_candidates` embeds the cleaned submission titles in a single local `fastembed` pass and queries PostgreSQL for the top-5 closest existing `Release` and `Artist` records per title ($\text{cosine similarity} \ge 0.50$). Deduplicated candidates are formatted into the prompt's `"Existing Catalog Reference"` block, and submissions within each batch are evaluated simultaneously by `ExtractionWorker.extract_batch` using a structured schema:

```python
class BatchExtractedItem(BaseModel):
    submission_id: str
    releases: list[ExtractedRelease]

class BatchExtractionResult(BaseModel):
    items: list[BatchExtractedItem]
```

### Fault Tolerance & Isolation
1. **Batch Exception Fallback**: If an entire batch request fails (network error, timeout, or schema parsing), the worker catches the error and immediately falls back to extracting each submission individually while preserving catalog candidate retrieval/context.
2. **Omission Fallback**: If the LLM omits one or more submission IDs in its response, missing submissions are automatically extracted via individual fallback with catalog candidate context. Any failure during individual extraction is safely handled without failing other submissions.
3. **Queue Pause on Shutdown**: When `SIGINT` or `SIGTERM` is received, running batches complete cleanly, and subsequent batches are paused until restart.

See [ADR 0006: Array Batch Extraction with Automated Single-Submission Fallback](docs/adr/0006-array-batch-extraction-with-fallback.md) and [ADR 0007: Pre-Retrieval Vector Context for Canonical Entity Alignment](docs/adr/0007-vector-canonical-alignment.md) for architectural details.

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
