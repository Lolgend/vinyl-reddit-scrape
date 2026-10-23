# Unpartitioned Snapshot Storage and Manual Retention Lifecycle

High-frequency engagement metric observations (`submission_snapshots`) accumulate continuously (~5,000 rows/day at ~15MB/month). Managing this growth could involve PostgreSQL declarative range partitioning or automated in-process retention cleanup jobs.

We decided to keep `submission_snapshots` as a standard unpartitioned PostgreSQL table indexed on `(submission_id, captured_at DESC)` and `captured_at`. Declarative range partitioning introduces unnecessary schema rigidity (composite primary keys) and maintenance overhead for a moderate data volume easily handled by standard B-tree indexes. 

Furthermore, the scraper application is strictly scoped to ingestion, metric observation, and structured extraction; no automated daily retention worker runs inside the scraping process. Snapshot retention pruning is decoupled as an on-demand manual operation. Each snapshot observation unconditionally appends to `submission_snapshots` to preserve uniform time-series density for downstream analytics, while updating `submissions` with the latest observed engagement metrics (`latest_score`, `latest_upvote_ratio`, `latest_num_comments`) so terminal engagement state is preserved permanently even after ephemeral snapshots are purged.
