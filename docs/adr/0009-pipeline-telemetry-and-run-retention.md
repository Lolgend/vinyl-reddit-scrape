---
status: accepted
---

# Pipeline Telemetry via Shared PostgreSQL Tables and Automatic Run Retention

To expose live **Pipeline** liveness, loop schedules, and **Pipeline Run** history to the strictly read-only `vinyl-reddit-app` `/admin` dashboard without adding inter-process networking or modifying existing submission/catalog schemas, `vinyl-reddit-scrape` owns and persists three operational telemetry tables (`pipeline_process_status`, `pipeline_loop_states`, `pipeline_runs`) in the shared PostgreSQL database. Telemetry writes are strictly isolated and defensively non-fatal so telemetry failures can never block or crash ingestion, extraction, or enrichment.

Because `pipeline_runs` is purely ephemeral operational telemetry rather than analytical time-series data, `PollerService.run()` automatically prunes expired `pipeline_runs` rows on startup and every 24 hours (in addition to exposing a manual CLI purge command). This intentionally scopes [ADR-0005](./0005-submission-snapshot-storage-and-retention.md)'s manual-only retention policy to `submission_snapshots`.

## Considered Options

- **Scraper-owned PostgreSQL telemetry tables with automatic run pruning (chosen)**: Keeps `vinyl-reddit-scrape` and `vinyl-reddit-app` decoupled via PostgreSQL, preserves `vinyl-reddit-app`'s read-only database boundary, leaves `submissions` and catalog tables untouched, and prevents unbounded growth of operational run logs without requiring an external cron job.
- **Direct HTTP/IPC status server on `vinyl-reddit-scrape`**: Adds port management, network coupling, and runtime availability dependencies between processes.
- **Manual-only retention for `pipeline_runs` (mirroring ADR-0005)**: Avoids an in-process cleanup call, but risks silent unbounded table growth in long-running scraper deployments for purely operational logs that carry no long-term analytical value.

## Consequences

- `vinyl-reddit-scrape` owns the schema DDL and ORM models (`PipelineProcessStatus`, `PipelineLoopState`, `PipelineRun`, and their enums) exported via `reddit_scrape.models` and top-level `reddit_scrape` for read-only import by `vinyl-reddit-app`.
- Assumes a single active **Pipeline** process (last-writer-wins keyed by `"default"`, with `host` and `pid` recorded for visibility) and no distributed locking.
- Because `submissions` intentionally carries no per-submission extraction error column, failed **Extraction Queue** drains surface batch/submission failures via loop-level `last_error` / `pipeline_runs` and apply capped in-memory exponential backoff (`10s` to `300s`) when zero items succeed to prevent 2-second DB/LLM write storms.
- Per-album **Catalog Enrichment** failures remain recorded on `albums.enrichment_error`; `enrichment_worker` loop-level `error` status is reserved for unhandled loop/DB exceptions.
