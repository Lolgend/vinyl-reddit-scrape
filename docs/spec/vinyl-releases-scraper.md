# Technical Specification: r/vinylreleases Ingestion & Structured Release Extraction Pipeline

## 1. Overview & System Architecture

The system discovers, extracts, and tracks vinyl record announcements posted to `r/vinylreleases`. It converts community-submitted unstructured posts into clean relational data using DeepSeek via `pydantic-ai`, and captures periodic time-series metric snapshots stored in PostgreSQL.

Downstream analytics (such as engagement velocity calculation, freshness decay scoring, and leaderboard generation) are explicitly decoupled from this ingestion pipeline and consume the historical time-series data captured in `submission_snapshots`.

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
    ValidateSchema --> T_SubRel

    T_Snapshots -.->|Time-Series Read| AnalyticsEngine
```

---

## 2. PostgreSQL Relational Schema

### DDL Definitions

```sql
-- 1. Raw Submissions
CREATE TABLE IF NOT EXISTS submissions (
    id VARCHAR(32) PRIMARY KEY, -- Reddit Fullname, e.g. 't3_1gxy7z' or '1gxy7z'
    reddit_id VARCHAR(16) NOT NULL UNIQUE,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    permalink TEXT NOT NULL,
    flair TEXT,
    author VARCHAR(64),
    is_self BOOLEAN DEFAULT FALSE,
    selftext TEXT,
    created_utc TIMESTAMPTZ NOT NULL,
    is_extracted BOOLEAN DEFAULT FALSE,
    is_release BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_submissions_created_utc ON submissions(created_utc DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_is_extracted ON submissions(is_extracted) WHERE is_extracted = FALSE;

-- 2. Submission Snapshots (Time-Series Metric Observations)
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
    purchase_url TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_releases_combined_artist ON releases(combined_artist);
CREATE INDEX IF NOT EXISTS idx_releases_album ON releases(album_title);

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

---

## 3. DeepSeek LLM Extraction Pipeline

### Configuration
Utilize PydanticAI's OpenAI-compatible model layer connected to DeepSeek's API endpoint:
- **Base URL**: `https://api.deepseek.com`
- **Model Name**: `deepseek-chat`
- **Environment Variable**: `DEEPSEEK_API_KEY`

### Structured Output Schema
```python
from typing import Optional
from pydantic import BaseModel, Field

class ExtractedRelease(BaseModel):
    artists: list[str] = Field(
        description="List of individual contributing artist or band names (e.g. ['Converge', 'Dropdead'] or ['Kanye West', 'Ty Dolla $ign'] or ['Charli XCX'])"
    )
    combined_artist: Optional[str] = Field(
        None,
        description="Combined billing display string when multiple artists collaborate (e.g. 'Converge / Dropdead', 'Kanye West & Ty Dolla $ign')"
    )
    album_title: str = Field(
        description="Title of the album, EP, single, or box set"
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

### System Prompt Design
```text
You are an expert vinyl discography classifier specialized in parsing posts from r/vinylreleases.

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
- "Restock: Artist - Album (Variant details)"
- "Artist - Album | Exclusive /500"

Your task:
1. Determine if the post announces a vinyl release (`is_vinyl_release`). Set to false if it is general discussion, a question, an in-search-of (ISO) request, or equipment sales.
2. Extract all distinct vinyl releases mentioned in the title.
3. Clean artist and album names by removing noise tags, bracketed dates, and parenthetical variant markers.
4. Categorize `release_type`: 'new_release', 'pre_order', 'restock', 'repress', 'bootleg', or 'unknown'.
5. Assign a confidence score from 0.0 to 1.0.
```

---

## 4. Ingestion Poller Specification

1. **Client**: Python `praw.Reddit` with `client_id`, `client_secret`, and descriptive `user_agent`.
2. **Target Endpoints**:
   - `reddit.subreddit("vinylreleases").new(limit=50)`: Polled every 5 minutes to capture new announcements immediately.
   - `reddit.subreddit("vinylreleases").hot(limit=50)`: Polled every 15 minutes to capture updated engagement metrics and track lifecycle flair updates.
3. **Filtering**:
   - Skip posts with explicit non-release flairs: `Discussion`, `Question`, `Collection`, `Advice`, `ISO`.
4. **Idempotency**:
   - Upsert `Submission` record on conflict (`ON CONFLICT (reddit_id) DO UPDATE ...`).
   - Append `SubmissionSnapshot` record on each observation cycle with current `score`, `upvote_ratio`, `num_comments`, and `flair`.

---

## 5. Polling Lifecycle & State Management

The ingestion pipeline runs continuously on two complementary polling schedules to balance immediate discovery with ongoing engagement metric tracking:

### 5.1 Discovery Polling (`r/vinylreleases/new`)
- **Interval**: Every 5 minutes (`POLL_INTERVAL_SECONDS = 300`).
- **Endpoint**: `reddit.subreddit("vinylreleases").new(limit=50)`.
- **Logic**:
  1. Filter out submissions with non-release flairs (`Discussion`, `Question`, `ISO`, etc.).
  2. Perform an idempotent upsert into `submissions`:
     ```sql
     INSERT INTO submissions (id, reddit_id, title, url, permalink, flair, author, is_self, selftext, created_utc, is_extracted, is_release)
     VALUES (:id, :reddit_id, :title, :url, :permalink, :flair, :author, :is_self, :selftext, :created_utc, FALSE, TRUE)
     ON CONFLICT (reddit_id) DO UPDATE SET
         flair = EXCLUDED.flair,
         updated_at = CURRENT_TIMESTAMP;
     ```
  3. If newly inserted, queue for LLM extraction (`is_extracted = FALSE`).
  4. Append initial metric observation to `submission_snapshots`:
     ```sql
     INSERT INTO submission_snapshots (submission_id, score, upvote_ratio, num_comments, flair, captured_at)
     VALUES (:submission_id, :score, :upvote_ratio, :num_comments, :flair, CURRENT_TIMESTAMP);
     ```

### 5.2 Active Tracking Polling (`r/vinylreleases/hot`)
- **Interval**: Every 15 minutes.
- **Endpoint**: `reddit.subreddit("vinylreleases").hot(limit=50)`.
- **Logic**:
  1. For active submissions appearing in `/hot`, update latest submission metadata:
     - Check if the submission flair has mutated (notably transitions to `SOLD OUT` or `EXPIRED`).
     - Update `submissions.flair` and `submissions.updated_at`.
  2. Append an updated metric observation to `submission_snapshots` with the current `score`, `upvote_ratio`, `num_comments`, and current `flair`.
  3. If an active hot submission was missed during previous `/new` cycles, insert it and queue for extraction.

### 5.3 Extraction Worker Lifecycle
- An asynchronous worker queries pending submissions:
  ```sql
  SELECT id, title, flair FROM submissions WHERE is_extracted = FALSE AND is_release = TRUE ORDER BY created_utc ASC LIMIT 10;
  ```
- Executes string pre-cleaning and passes the cleaned title/flair to DeepSeek (`deepseek-chat`) via PydanticAI.
- In a single transaction:
  - Inserts extracted `releases` records.
  - Links them to `submissions` via `submission_releases`.
  - Sets `is_extracted = TRUE` on the submission record.

---

## 6. Configuration Environment (`.env`)

```bash
# Reddit API (Script App)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=python:vinyl-releases-scraper:v0.1.0 (by /u/username)

# Database
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/vinyl_releases

# LLM (DeepSeek)
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

# Polling Intervals
POLL_INTERVAL_SECONDS=300
HOT_POLL_INTERVAL_SECONDS=900
SNAPSHOT_RETENTION_DAYS=30
```
