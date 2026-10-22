# Technical Specification: r/vinylreleases Ingestion & Hype Intelligence Engine

## 1. Overview & System Architecture

The system discovers, extracts, tracks, and ranks vinyl record announcements posted to `r/vinylreleases`. It converts community-submitted unstructured posts into clean relational data using DeepSeek via `pydantic-ai`, measures community engagement over time using periodic time-series snapshots, and computes an engagement velocity hype score stored in PostgreSQL.

```mermaid
flowchart TD
    subgraph Reddit["Reddit API"]
        SubNew["r/vinylreleases/new"]
        SubHot["r/vinylreleases/hot"]
    end

    subgraph Poller["Ingestion Poller (PRAW)"]
        Fetch["Fetch Submissions (Every 5-15 min)"]
        FilterFlairs{"Check Flair\n(Exclude Discussion/ISO)"}
    end

    subgraph Database["PostgreSQL Storage"]
        T_Submissions[("submissions")]
        T_Snapshots[("submission_snapshots")]
        T_Releases[("releases")]
        T_SubRel[("submission_releases")]
        T_Hype[("hype_metrics")]
    end

    subgraph Extractor["LLM Extraction Layer (DeepSeek + PydanticAI)"]
        PreClean["Pre-cleaning (Strip tags, markdown)"]
        DeepSeekAgent["DeepSeek Chat Agent\n(Structured Output)"]
        ValidateSchema["Pydantic Validation (ExtractedRelease)"]
    end

    subgraph HypeEngine["Hype Velocity Calculator"]
        DiffCalc["Calculate Velocity (ΔScore/Δt, ΔComments/Δt)"]
        DecayCalc["Apply Freshness Decay"]
        RankTable["Update Hype Leaderboard"]
    end

    SubNew --> Fetch
    SubHot --> Fetch
    Fetch --> FilterFlairs
    FilterFlairs -->|Valid| T_Submissions
    Fetch --> T_Snapshots

    T_Submissions -->|Unextracted Submission| PreClean
    PreClean --> DeepSeekAgent
    DeepSeekAgent --> ValidateSchema
    ValidateSchema --> T_Releases
    ValidateSchema --> T_SubRel

    T_Snapshots --> DiffCalc
    DiffCalc --> DecayCalc
    DecayCalc --> T_Hype
    T_Hype --> RankTable
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
    captured_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_snapshots_sub_time ON submission_snapshots(submission_id, captured_at DESC);

-- 3. Normalized Releases
CREATE TABLE IF NOT EXISTS releases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artist VARCHAR(255) NOT NULL,
    album_title VARCHAR(255) NOT NULL,
    format_variant TEXT,
    retailer_name VARCHAR(255),
    purchase_url TEXT,
    release_type VARCHAR(32) NOT NULL DEFAULT 'new_release',
    confidence_score FLOAT NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_releases_artist ON releases(artist);
CREATE INDEX IF NOT EXISTS idx_releases_album ON releases(album_title);

-- 4. Many-to-Many Association
CREATE TABLE IF NOT EXISTS submission_releases (
    submission_id VARCHAR(32) NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    release_id UUID NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (submission_id, release_id)
);

-- 5. Calculated Hype Metrics
CREATE TABLE IF NOT EXISTS hype_metrics (
    submission_id VARCHAR(32) PRIMARY KEY REFERENCES submissions(id) ON DELETE CASCADE,
    current_score INT NOT NULL DEFAULT 0,
    current_comments INT NOT NULL DEFAULT 0,
    upvote_velocity_1h FLOAT NOT NULL DEFAULT 0.0,
    comment_velocity_1h FLOAT NOT NULL DEFAULT 0.0,
    hype_index FLOAT NOT NULL DEFAULT 0.0,
    last_calculated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_hype_index ON hype_metrics(hype_index DESC);
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
from typing import Literal, Optional
from pydantic import BaseModel, Field

class ExtractedRelease(BaseModel):
    artist: str = Field(description="Name of the musical artist or group")
    album_title: str = Field(description="Name of the album, EP, single, or soundtrack")
    format_variant: Optional[str] = Field(
        None, description="Color pressing, vinyl weight, limitation (e.g., 'Pink Swirl /500', '180g 2LP')"
    )
    retailer: Optional[str] = Field(
        None, description="Retailer, exclusive seller, or record label (e.g., 'Blood Records', 'Target', 'Rough Trade')"
    )
    release_type: Literal['new_release', 'pre_order', 'restock', 'repress', 'bootleg', 'unknown'] = Field(
        'new_release', description="The release event indicated by the title or flair"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the extraction accuracy from 0.0 to 1.0"
    )

class ExtractionResult(BaseModel):
    is_vinyl_release: bool = Field(
        description="Whether this submission represents an actual vinyl release announcement or drop"
    )
    releases: list[ExtractedRelease] = Field(
        default_factory=list, description="List of releases extracted from the submission title"
    )
```

### System Prompt Design
```text
You are an expert vinyl discography classifier specialized in parsing posts from r/vinylreleases.
Subreddit titles follow patterns such as:
- "[Pre-Order] [Release Date] Artist - Album (Format / Color / Pressing count) [Store Name]"
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
   - `reddit.subreddit("vinylreleases").new(limit=50)`: Captured on every cycle to catch new announcements immediately.
   - `reddit.subreddit("vinylreleases").hot(limit=30)`: Captured to update active engagement snapshots for high-activity drops.
3. **Filtering**:
   - Skip posts with explicit non-release flairs: `Discussion`, `Question`, `Collection`, `Advice`, `ISO`.
4. **Idempotency**:
   - Upsert `Submission` record on conflict (`ON CONFLICT (reddit_id) DO UPDATE ...`).
   - Append `SubmissionSnapshot` record with current `score`, `upvote_ratio`, `num_comments`.

---

## 5. Hype Velocity & Analytics Engine

### 5.1 Velocity Calculation
Given two snapshots $S_1 = (t_1, \text{score}_1, \text{comments}_1)$ and $S_2 = (t_2, \text{score}_2, \text{comments}_2)$ where $\Delta t = t_2 - t_1$ in hours:

$$\text{Upvote Velocity } V_u = \max\left(0, \frac{\text{score}_2 - \text{score}_1}{\Delta t}\right)$$

$$\text{Comment Velocity } V_c = \max\left(0, \frac{\text{comments}_2 - \text{comments}_1}{\Delta t}\right)$$

### 5.2 Freshness Decay Factor
To prevent older, permanently popular posts from dominating the live leaderboard over newly breaking releases, apply a power decay based on post age $t_{\text{age}}$ (in hours):

$$D(t) = \frac{1}{\left(1 + \frac{t_{\text{age}}}{24}\right)^{1.3}}$$

### 5.3 Compound Hype Index
$$H = \left(w_u \cdot V_u + w_c \cdot V_c \cdot (1 + \log_{10}(1 + \text{comments}_2))\right) \cdot D(t)$$
*Default Weights*: $w_u = 1.0$, $w_c = 2.5$ (comments reflect active community discussion and variant demand).

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
SNAPSHOT_RETENTION_DAYS=30
```
