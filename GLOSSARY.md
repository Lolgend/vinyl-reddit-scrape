# Vinyl Releases Scraper Domain

Domain vocabulary for scraping, extracting, tracking, and ranking vinyl record releases from community submissions.

## Language

**Submission**:
A raw post imported from the `r/vinylreleases` subreddit representing a community-submitted announcement or link.
_Avoid_: Post, thread, article, listing

**Submission Snapshot**:
A point-in-time timestamped observation of a submission's engagement metrics (score, upvote ratio, comment count).
_Avoid_: Metric check, poll log, vote record

**Extraction Batch**:
A bounded chunk of unextracted candidate submissions (default: 5) evaluated simultaneously in a single LLM prompt with automated fallback to single-submission extraction on failure.
_Avoid_: Request chunk, extraction group, bulk prompt

**Release**:
A distinct vinyl record product or edition announced to the public, identified by artist(s), album title, variant, and retailer.
_Avoid_: Record, vinyl, drop, product, item

**Artist**:
An individual musical creator, performer, composer, or band contributing to a release. Multi-artist releases associate each contributing artist individually alongside a combined billing.
_Avoid_: Creator, musician, singer

**Combined Artist**:
The collaborative or split billing representation of multiple artists on a single release (e.g., `Converge / Dropdead` or `Kanye West & Ty Dolla $ign`).
_Avoid_: Dual artist, joint name, multi-band

**Variant**:
A specific physical pressing variation of a release, such as color pressing, weight (e.g., 180g), or limited edition count.
_Avoid_: Edition, colorway, version

**Release Event**:
The type of commercial availability for a release, categorized as new release, pre-order, restock, or repress.
_Avoid_: Status, category, drop type

**Hype Velocity**:
The instantaneous or short-window rate of change of community engagement metrics ($\Delta \text{score}/\Delta t$, $\Delta \text{comments}/\Delta t$).
_Avoid_: Popularity speed, vote rate, hype momentum

**Hype Index**:
A normalized composite score that balances engagement velocity with freshness decay over time.
_Avoid_: Trending score, rank score, hotness
