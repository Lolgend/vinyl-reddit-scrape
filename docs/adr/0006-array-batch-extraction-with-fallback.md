# Array Batch Extraction with Automated Single-Submission Fallback

## Context

Sequential single-submission extraction issues an individual LLM prompt and API round-trip for every discovered release candidate. While accurate, sequential processing introduces high latency (~1-2 seconds per submission) and repetitive prompt token overhead (system prompts, discography guidelines, and schema instructions sent once per submission). During high-volume poll windows or ingestion backlogs, single-submission queues bottleneck extraction throughput and increase operational costs.

We evaluated batch extraction techniques to optimize throughput without compromising extraction accuracy, entity isolation, or transaction resilience.

## Decision

We decided to implement chunked array batch extraction in `ExtractionWorker` and `PollerService` with automated single-submission fallback:

1. **Chunked Queue Processing**:
   Unextracted candidate submissions (`is_extracted=False`, `is_release=True`) are queried in FIFO order by `created_utc` and partitioned into bounded chunks of size `llm_batch_size` (default: 5, configurable via `LLM_BATCH_SIZE` or `--llm-batch-size`).

2. **Structured 1-to-1 Submission Association Schema**:
   Batches are evaluated using `BatchExtractionResult` containing `items: list[BatchExtractedItem]`, where each item binds a unique Reddit `submission_id` to its extracted `releases: list[ExtractedRelease]`. This guarantees deterministic 1-to-1 entity association without title ambiguity or release cross-contamination across posts.

3. **Two-Tier Automated Fallback**:
   - **Batch Failure Fallback**: If an entire batch LLM call fails (network disruption, model timeout, or schema parsing error), `ExtractionWorker.extract_batch` catches the exception and immediately falls back to extracting each submission individually via single-submission `extract()`.
   - **Omission Fallback**: If the LLM generates a valid batch response but omits one or more submission IDs, the missing submissions are detected and extracted individually via single-submission `extract()`.

4. **Performance, Cost, and Chunk Size Trade-offs**:
   - A default chunk size of 5 submissions was selected based on empirical benchmarking. At 5 submissions per batch, prompt token overhead is amortized across multiple items without exceeding LLM output token limits or degrading reasoning quality.
   - Larger chunks (>10 submissions) increase the risk of output token truncation, omitted submission IDs, and higher latency per individual batch request.
   - Fallback isolation ensures that a single malformed post or schema anomaly within a chunk cannot prevent the remaining submissions in the chunk or queue from being persisted.
