# Time-Series Snapshots for Ingestion & Decoupled Downstream Scoring

Newly announced releases initially have zero comments and a single upvote, making a one-time static metric check ineffective for detecting real community momentum or stock status mutations (e.g. `SOLD OUT`). 

We decided to capture periodic timestamped snapshots of engagement metrics (`submission_snapshots`) within the ingestion pipeline to preserve historical time-series data. However, calculating derived engagement velocity ($\Delta \text{score}/\Delta t$, $\Delta \text{comments}/\Delta t$), freshness decay parameterization, and leaderboard ranking windows is decoupled from the scraper engine and deferred to downstream analytics/backend services. This maintains a clean single responsibility for the scraper: high-fidelity ingestion, raw metric logging, and structured release extraction.

### Amendment (Spec #22): Snapshot Schema & Dynamic Availability Tracking

Per Spec #22 and the domain glossary definition, `SubmissionSnapshot` records strictly quantitative point-in-time engagement observations (`score`, `upvote_ratio`, `num_comments`, `captured_at`). Dynamic availability/status mutations (e.g. `SOLD OUT`) are recorded and updated in place on parent `Submission.flair` rather than duplicated across high-frequency time-series snapshot rows.
