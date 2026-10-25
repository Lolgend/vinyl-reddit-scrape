# Technical Specification: r/vinylreleases Ingestion & Structured Release Extraction Pipeline

## Problem Statement

Vinyl collectors and enthusiasts lack a reliable, structured way to discover and track new vinyl record releases announced on `r/vinylreleases`. Submissions are posted by community members with inconsistent title formatting, noisy tags, varied delimiters, and transient flairs (e.g., dynamically mutating to `SOLD OUT`). Furthermore, popular releases are frequently posted multiple times across different retail storefronts. Without an automated ingestion and structured entity extraction pipeline, collectors cannot filter out discussions, track pressing variants, or capture engagement momentum.

## Solution

An automated ingestion and extraction pipeline that:
1. Polls `r/vinylreleases` on dual polling intervals (5 minutes for discovery on `/new`, 15 minutes for tracking on `/hot`).
2. Persists raw submissions and appends unconditional point-in-time engagement snapshots to PostgreSQL.
3. Extracts structured release entities (individual artists, combined artist billing, album title, physical variant, and exclusive retailer) using DeepSeek LLM via PydanticAI.
4. Uniquely identifies and deduplicates multi-submission releases via a deterministic SHA-256 metadata fingerprint.
5. Preserves terminal engagement metrics on parent submissions while supporting decoupled, on-demand snapshot retention pruning.

```mermaid
flowchart TD
    subgraph Reddit["Reddit API (r/vinylreleases)"]
        SubNew["r/vinylreleases/new (5m poll)"]
        SubHot["r/vinylreleases/hot (15m poll)"]
    end

    subgraph Poller["Ingestion Poller (PRAW)"]
        FetchNew["Fetch New Submissions"]
        FetchHot["Fetch Active / Hot Submissions"]
        FilterFlairs{"Check Flair\n(Exclude Discussion/ISO)"}
    end

    subgraph Database["PostgreSQL Storage"]
        T_Submissions[("submissions")]
        T_Snapshots[("submission_snapshots")]
        T_Releases[("releases")]
        T_SubRel[("submission_releases")]
        T_Artists[("artists")]
        T_RelArt[("release_artists")]
    end

    subgraph Extractor["LLM Extraction Layer (DeepSeek + PydanticAI)"]
        Queue["Unextracted Queue\n(is_extracted = FALSE)"]
        PreClean["Pre-cleaning (Strip tags, normalize dashes)"]
        DeepSeekAgent["DeepSeek Chat Agent\n(Structured Output)"]
        ValidateSchema["Pydantic Validation (ExtractedRelease)"]
    end

    subgraph Downstream["Downstream Backend / Analytics (Decoupled)"]
        AnalyticsEngine["Velocity, Decay & Leaderboards"]
    end

    SubNew --> FetchNew
    SubHot --> FetchHot
    FetchNew --> FilterFlairs
    FilterFlairs -->|Valid Release Candidate| T_Submissions
    FilterFlairs -->|Initial Snapshot| T_Snapshots

    FetchHot -->|Update Metadata / Flair| T_Submissions
    FetchHot -->|Append Timestamped Metric| T_Snapshots

    T_Submissions --> Queue
    Queue --> PreClean
    PreClean --> DeepSeekAgent
    DeepSeekAgent --> ValidateSchema
    ValidateSchema --> T_Releases
    ValidateSchema --> T_Artists
    ValidateSchema --> T_RelArt
    ValidateSchema --> T_SubRel

    T_Snapshots -.->|Time-Series Read| AnalyticsEngine
```

## User Stories

1. As a vinyl collector, I want the system to poll `r/vinylreleases` every 5 minutes, so that I discover newly announced vinyl releases as soon as they are submitted.
2. As a vinyl collector, I want the system to extract structured artist and album names from submission titles, so that I can browse releases cleanly without subreddit title noise.
3. As a vinyl collector, I want multi-artist collaborations or splits to be separated into individual artist entities, so that I can follow individual creators across joint projects.
4. As a vinyl collector, I want collaborative releases to maintain an official combined artist billing, so that the displayed release name reflects official packaging.
5. As a vinyl collector, I want physical pressing variants (e.g., colorways, limited run counts, vinyl weight) extracted into a dedicated field, so that I know exactly which edition is announced.
6. As a vinyl collector, I want exclusive retailers or record labels identified, so that I know where the release is commercially sold.
7. As a vinyl collector, I want non-release submissions (such as discussions, questions, and ISO requests) filtered out, so that non-release posts do not pollute the catalog.
8. As a vinyl collector, I want generic store restock announcements lacking specific album titles to return empty releases, so that vague sales do not create junk database entries.
9. As a vinyl collector, I want multiple submissions announcing the same release to link to a single canonical release record, so that duplicate community posts do not fragment the release catalog.
10. As a vinyl collector, I want deduplication to function accurately even when submissions link to different retail URLs or artist homepages, so that identical pressings are properly consolidated.
11. As a data analyst, I want the system to log periodic timestamped submission snapshots of score, upvote ratio, and comment count, so that I can calculate engagement velocity and freshness decay downstream.
12. As a data analyst, I want active hot submissions polled every 15 minutes, so that trending releases have high-frequency metric tracking during their peak hype window.
13. As a data analyst, I want snapshots recorded unconditionally on every poll cycle, so that time-series analysis has uniform data density without gaps caused by delta suppression.
14. As a data analyst, I want submissions to track dynamic flair mutations (e.g., transitions to "SOLD OUT"), so that downstream models can detect stock exhaustion events.
15. As a data analyst, I want terminal engagement metrics preserved on parent submission records permanently, so that final community reception remains queryable even after ephemeral snapshots are purged.
16. As a pipeline operator, I want submissions upserted idempotently by Reddit ID, so that repeated polling cycles never create duplicate submission records or fail on database conflicts.
17. As a pipeline operator, I want release deduplication to use a deterministic SHA-256 fingerprint generated from normalized metadata, so that release identity is consistent, reproducible, and enforced by database unique constraints.
18. As a pipeline operator, I want missing variants or retailers to use explicit standard tokens in the fingerprint calculation, so that standard editions never falsely collide with limited variants or retailer exclusives.
19. As a pipeline operator, I want LLM extraction to run asynchronously on unextracted submissions, so that Reddit polling ingestion is never blocked by LLM API latency.
20. As a pipeline operator, I want all release, artist, and join table updates for a submission to commit in a single database transaction, so that extraction failures do not leave orphaned or partial records.
21. As a pipeline operator, I want snapshot pruning to be an on-demand administrative operation rather than an automated in-process background worker, so that the scraper application remains strictly focused on ingestion and extraction.
22. As a pipeline operator, I want snapshot pruning to use batched deletions, so that high-volume maintenance jobs do not hold long-lived table locks or degrade ingestion performance.
23. As a developer, I want the LLM extraction agent to use DeepSeek via PydanticAI with strict anti-hallucination prompting, so that missing artists or albums are never hallucinated.
24. As a developer, I want pre-cleaning logic to normalize unicode dashes and whitespace before LLM extraction, so that input text noise is eliminated before tokenization.

## Implementation Decisions

### 1. Ingestion Poller & Polling Lifecycle
- **Discovery Polling (`/new`)**: Runs every 5 minutes (`POLL_INTERVAL_SECONDS = 300`) fetching the top 50 new submissions.
- **Tracking Polling (`/hot`)**: Runs every 15 minutes (`HOT_POLL_INTERVAL_SECONDS = 900`) fetching top 50 active submissions to monitor score changes and lifecycle flair transitions (e.g., `SOLD OUT`, `EXPIRED`).
- **Flair Filtering**: Immediately filters out posts tagged with `Discussion`, `Question`, `Collection`, `Advice`, or `ISO`.
- **Submission Upsert**:
  ```sql
  INSERT INTO submissions (id, reddit_id, title, url, flair, latest_score, latest_upvote_ratio, latest_num_comments, created_utc, is_extracted, is_release)
  VALUES (:id, :reddit_id, :title, :url, :flair, :score, :upvote_ratio, :num_comments, :created_utc, FALSE, TRUE)
  ON CONFLICT (reddit_id) DO UPDATE SET
      flair = EXCLUDED.flair,
      latest_score = EXCLUDED.latest_score,
      latest_upvote_ratio = EXCLUDED.latest_upvote_ratio,
      latest_num_comments = EXCLUDED.latest_num_comments,
      updated_at = CURRENT_TIMESTAMP;
  ```
- **Unconditional Snapshot Logging**: Appends metric observations on each observation cycle:
  ```sql
  INSERT INTO submission_snapshots (submission_id, score, upvote_ratio, num_comments, flair, captured_at)
  VALUES (:submission_id, :score, :upvote_ratio, :num_comments, :flair, CURRENT_TIMESTAMP);
  ```

### 2. PostgreSQL Relational Storage & DDL
The database defines 6 normalized tables:
```sql
-- 1. Raw Submissions
CREATE TABLE IF NOT EXISTS submissions (
    id VARCHAR(32) PRIMARY KEY,
    reddit_id VARCHAR(16) NOT NULL UNIQUE,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    flair TEXT,
    latest_score INT DEFAULT 0,
    latest_upvote_ratio FLOAT DEFAULT 1.0,
    latest_num_comments INT DEFAULT 0,
    created_utc TIMESTAMPTZ NOT NULL,
    is_extracted BOOLEAN DEFAULT FALSE,
    is_release BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_submissions_created_utc ON submissions(created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_is_extracted ON submissions(is_extracted) WHERE is_extracted = FALSE;

-- 2. Submission Snapshots (Unpartitioned Time-Series Metrics)
CREATE TABLE IF NOT EXISTS submission_snapshots (
    id BIGSERIAL PRIMARY KEY,
    submission_id VARCHAR(32) NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    score INT NOT NULL,
    upvote_ratio FLOAT NOT NULL,
    num_comments INT NOT NULL,
    flair TEXT,
    captured_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_snapshots_sub_time ON submission_snapshots(submission_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_captured_at ON submission_snapshots(captured_at);

-- 3. Normalized Artists
CREATE TABLE IF NOT EXISTS artists (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_artists_name ON artists(name);

-- 4. Normalized Releases
CREATE TABLE IF NOT EXISTS releases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    combined_artist VARCHAR(255) NOT NULL,
    album_title VARCHAR(255) NOT NULL,
    variant TEXT,
    retailer_name VARCHAR(255),
    fingerprint VARCHAR(64) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_releases_combined_artist ON releases(combined_artist);
CREATE INDEX IF NOT EXISTS idx_releases_album ON releases(album_title);
CREATE UNIQUE INDEX IF NOT EXISTS idx_releases_fingerprint ON releases(fingerprint);

-- 5. Release-Artist Many-to-Many Association
CREATE TABLE IF NOT EXISTS release_artists (
    release_id UUID NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    artist_id UUID NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (release_id, artist_id)
);

-- 6. Submission-Release Many-to-Many Association
CREATE TABLE IF NOT EXISTS submission_releases (
    submission_id VARCHAR(32) NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    release_id UUID NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (submission_id, release_id)
);
```

### 3. DeepSeek LLM Extraction Layer
Structured entity extraction uses DeepSeek (`deepseek-chat`) via PydanticAI with `OpenAIProvider`. Inlined schema validated from prototype:
```python
from typing import Optional
from pydantic import BaseModel, Field

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
```

System prompt enforces anti-hallucination:
```text
You are an expert vinyl record discography classifier specialized in extracting structured data from r/vinylreleases submission titles.

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
```

### 4. Deterministic Identity & Deduplication
- Computed as a deterministic SHA-256 fingerprint:
  `sha256(f"{norm_artist}|{norm_album}|{norm_variant}|{norm_retailer}")`
- Normalization:
  - Lowercases and collapses whitespace.
  - Normalizes unicode dashes (`\u2010-\u2015\u2212` to `-`).
  - Missing `variant` or `retailer` is mapped to `[standard]`, preventing collisions between standard releases and retailer exclusives.
- Upsert execution:
  ```sql
  INSERT INTO releases (combined_artist, album_title, variant, retailer_name, fingerprint)
  VALUES (:combined_artist, :album_title, :variant, :retailer, :fingerprint)
  ON CONFLICT (fingerprint) DO UPDATE SET updated_at = CURRENT_TIMESTAMP
  RETURNING id;
  ```
- Omission of `purchase_url`: URL-based deduplication is rejected because links often point to homepage root domains or affiliate hubs. Multiple submissions link to the release via `submission_releases`.

### 5. Snapshot Retention & Purge Policy
- Unconditional metric observation per poll cycle guarantees uniform time-series density.
- The scraper runtime contains **no** automated background daily deletion worker.
- Retention cleanup is decoupled as a manual on-demand administrative maintenance script:
  ```sql
  DELETE FROM submission_snapshots 
  WHERE id IN (
      SELECT id FROM submission_snapshots 
      WHERE captured_at < NOW() - (INTERVAL '1 day' * :retention_days)
      LIMIT 5000
  );
  ```
- Submissions permanently retain terminal metrics (`latest_score`, `latest_upvote_ratio`, `latest_num_comments`).

## Testing Decisions

- **Good Test Criteria**: Tests verify external behavior through public module interfaces, observing database state transitions and validated schema objects rather than implementation details or SQL syntax.
- **Modules Under Test**:
  - **Pipeline Ingestion Service (Primary Seam)**: Tested using mock Reddit submission streams and a local PostgreSQL test database to verify end-to-end upserts, snapshot logging, extraction queuing, and transaction rollbacks.
  - **Extraction Agent (Secondary Seam)**: Tested against a diverse fixture catalog of r/vinylreleases post titles to verify regex pre-cleaning, schema parsing, and anti-hallucination behavior.
- **Prior Art**: Interactive extraction prototype (`prototype_extractor.py`) and validated fixture results in `prototype_results.json`.

## Out of Scope

- Downstream engagement velocity calculation, freshness decay weighting, and leaderboard ranking windows (decoupled per ADR 0002).
- Web frontend, graphical user dashboard, or collector UI.
- Automated checkout bots, stock monitoring scrapers, or cart automation.
- Real-time alerting or notification webhooks (Discord, Telegram, Email).
- Automated daily in-process background retention pruning workers (per ADR 0005).

## Further Notes

- **Configuration Environment**:
  - `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
  - `DATABASE_URL` (e.g. `postgresql+psycopg://postgres:password@localhost:5432/vinyl`)
  - `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` (`https://api.deepseek.com`), `DEEPSEEK_MODEL` (`deepseek-chat`)
  - `POLL_INTERVAL_SECONDS` (300), `HOT_POLL_INTERVAL_SECONDS` (900), `SNAPSHOT_RETENTION_DAYS` (30)
- **Windows UTF-8**: Windows environments must ensure `PYTHONUTF8=1` is set to avoid `cp1252` encoding exceptions on unicode dashes and special artist characters.
