# Technical Specification: Extraction Batching, CLI Verbosity, Thinking Level & Prototype Cleanup

## Problem Statement

When operating the Reddit Vinyl Releases scraper pipeline, operators and developers encounter several limitations:
1. **Silent Execution & Lack of Observability**: Running the ingestion poller executes silently without real-time console feedback on discovered submissions, poll loop cadences, or extraction results. Developers cannot easily monitor ingestion health or debug extraction failures without modifying code.
2. **Sequential Extraction Latency**: Unextracted submissions are processed individually one-by-one against the LLM extraction worker. When discovering surges of new vinyl releases, sequential API calls cause bottlenecked queue draining and unnecessary prompt token overhead.
3. **Unconfigurable Reasoning Overhead**: When deploying reasoning-capable models (e.g., DeepSeek reasoner or thinking-enabled variants), operators cannot configure the thinking level. Deep reasoning mode introduces high token costs and multi-second latencies that are excessive for structured discography parsing.
4. **Prototype Debt & Fixture Pollution**: Historical prototype scripts and demo artifacts from early spikes remain in the production package namespace, while regression test suites directly import fixtures from the source directory rather than a dedicated test fixture hierarchy.
5. **Missing CLI Documentation**: The project lacks user-facing CLI and operational documentation, leaving developers to read internal source code to understand configuration variables, execution modes, and arguments.

## Solution

An enhanced operational and extraction layer that:
1. Provides real-time console observability through standard streaming logging with `--verbose` (`-v`) and `--log-level` flags, surfacing poll counts, submission titles, and extraction summaries.
2. Implements chunked array batch extraction (defaulting to 5 submissions per prompt) with an automated single-submission fallback on schema validation or network error, achieving high throughput while guaranteeing zero data loss.
3. Exposes an explicit LLM thinking level configuration (`--thinking-level` / `LLM_THINKING_LEVEL`) defaulting to the lowest setting (`low` / `none`) across the PydanticAI model settings.
4. Cleans up obsolete prototype files from the production package, relocating benchmark datasets into a clean test fixture structure.
5. Delivers a comprehensive project `README.md` detailing architecture, environment configuration, and CLI commands.

---

## User Stories

1. As an operator running the scraper CLI, I want to pass a `-v` or `--verbose` flag so that I can see live timestamped logs of polling activity, fetched submission titles, and extraction results on my terminal.
2. As an operator troubleshooting an issue, I want to specify an explicit `--log-level` (DEBUG, INFO, WARNING, ERROR) so that I can inspect low-level network activity or suppress routine informational logs.
3. As a developer running single poll cycles with `--once`, I want verbose output to display individual discovered submissions and extraction timings so that I can immediately verify end-to-end functionality.
4. As an operator processing a high volume of new submissions, I want the extraction worker to batch multiple submissions in a single LLM prompt so that queue processing latency and API round-trip overhead are drastically reduced.
5. As an operator, I want to configure the batch size per LLM prompt via `--llm-batch-size` and environment variable `LLM_BATCH_SIZE` so that I can tune throughput based on provider rate limits and context constraints.
6. As a system architect, I want the extraction worker to enforce a bounded default batch size of 5 submissions so that LLM attention dispersion, output token truncation, and context overflow are avoided.
7. As a data pipeline engineer, I want the extraction worker to automatically fall back to single-submission extraction if a batch call fails schema validation or encounters a network error, so that transient batch errors never result in lost release data.
8. As a developer, I want the extraction worker to verify that every submission ID provided in a batch input is returned in the output payload, falling back to single-submission extraction for any omitted IDs.
9. As an operator managing LLM inference costs and latency, I want to configure the model's thinking/reasoning effort via `--thinking-level` and `LLM_THINKING_LEVEL` so that the model does not spend unnecessary reasoning tokens on structured discography extraction.
10. As an operator, I want the thinking level to default to the lowest setting (`low` for reasoning models, `none` for standard chat models) so that the scraper achieves optimal extraction speed and lowest cost out of the box.
11. As a downstream analytics engineer, I want raw submission `score` and `upvote_ratio` preserved as the canonical engagement metrics without calculating or storing derived vote counts during scraping, so that downstream models can dynamically evaluate momentum and decay.
12. As a maintainer, I want prototype scripts (`prototype_extractor.py`, `prototype_extractor_demo.html`) removed from the production package so that the codebase remains clean and maintainable.
13. As a test engineer, I want benchmark test datasets relocated from the source package into a dedicated `tests/fixtures/` directory so that test assets follow standard repository conventions.
14. As a test engineer, I want all existing and new unit tests to pass cleanly without regressions across extraction, ingestion, persistence, and CLI runner suites.
15. As a newcomer or developer joining the project, I want a comprehensive `README.md` explaining prerequisites, configuration variables, CLI usage, and test commands so that I can get started quickly.

---

## Implementation Decisions

### 1. Engagement Metric Integrity & Decoupled Downstream Scoring
In accordance with ADR 0002, the scraper strictly stores raw observation metrics (`score`, `upvote_ratio`, `num_comments`) without storing derived upvote/downvote approximations. Calculating velocity, decay, and momentum is decoupled from ingestion and reserved for downstream analytical engines.

### 2. Array Batch Extraction Schema & Chunking Strategy
Submissions are presented to the extraction worker in bounded chunks (default: 5 submissions per prompt). The LLM output schema guarantees 1-to-1 association by pairing each submission ID with its extracted release entities:

```python
class BatchExtractedItem(BaseModel):
    submission_id: str
    releases: list[ExtractedRelease]

class BatchExtractionResult(BaseModel):
    items: list[BatchExtractedItem]
```

If the batch invocation encounters a schema validation error, JSON decode error, or omits an input submission ID, the system intercepts the exception and dispatches single-submission extraction calls for the affected items.

### 3. PydanticAI Model Settings for Thinking Controls
The extraction worker configures reasoning effort via PydanticAI model settings using the unified thinking configuration parameter. The supported levels are `none`, `minimal`, `low`, `medium`, `high`, and `max`. The default value is set to `low` (or `none` when disabled), ensuring minimal latency and token consumption.

### 4. Standardized Stream Logging & CLI Flags
The CLI entrypoint configures the standard Python `logging` root handler when `-v` or `--verbose` is provided, setting the log level to `INFO` (or the level specified by `--log-level`). Formatted log output streams timestamps, log levels, logger names, and contextual operational events to stdout/stderr.

### 5. Prototype Decommissioning & Fixture Relocation
Experimental prototype scripts are removed from `src/reddit_scrape/`. The benchmark JSON dataset is relocated to `tests/fixtures/benchmark_cases.json`, and regression test suites are updated to load fixtures from the dedicated test path.

### 6. Comprehensive CLI Documentation
A root `README.md` documents environment variables, installation instructions, database setup, and all CLI subcommands (`run` and `purge-snapshots`) with detailed flag definitions and examples.

---

## Testing Decisions

### What Makes a Good Test
Tests must evaluate external behavior rather than private implementation details:
- Verify that passing valid submission tuples to the batch extraction interface produces correctly keyed release entities.
- Verify that simulated batch extraction failures gracefully trigger individual extractions and return complete results.
- Verify that unextracted candidate submissions are sliced into batches respecting configured chunk sizes.
- Verify that CLI arguments are correctly parsed and configure expected service and logging parameters.
- Verify that benchmark fixtures in `tests/fixtures/` pass schema conformance tests.

### Modules Tested
- Extraction worker and batch extraction schema conformance.
- Continuous poller service batch execution and unextracted candidate processing.
- CLI argument parsing, logging initialization, and runner integration.

### Prior Art
- Unit tests in `tests/test_extraction.py` for mocking PydanticAI agent runs.
- Integration tests in `tests/test_service.py` for mocking database sessions and service execution loops.

---

## Out of Scope

- Calculating or persisting derived upvote and downvote counts.
- Automated background retention daemons (retained as an on-demand administrative CLI command per ADR 0005).
- Modifying PostgreSQL database tables or migrations for submissions and releases.

---

## Further Notes

- Maintains full compatibility with ADR 0001 (Decouple Submissions and Releases), ADR 0002 (Time-Series Snapshots & Decoupled Downstream Scoring), and ADR 0003 (DeepSeek LLM Structured Release Extraction).
- Accompanied by ADR 0006 (Array Batch Extraction with Automated Fallback) and domain additions in `GLOSSARY.md`.
