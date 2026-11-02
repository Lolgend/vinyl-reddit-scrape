# Deterministic Fingerprint for Album and Release Identity and Deduplication

When multiple Reddit submissions announce the same vinyl release, deduplicating via retail URLs is unreliable because submissions frequently link to general storefronts, label hubs, or artist homepages rather than specific product SKUs. Furthermore, purchasing/procurement links are out of scope for the scraper pipeline.

We decided to uniquely identify musical works (`albums`) and physical pressing variants (`releases`) in PostgreSQL using deterministic SHA-256 `fingerprint` columns:
- **`Album.fingerprint`**: `sha256("{norm_combined_artist}|{norm_title}")` over normalized work metadata.
- **`Release.fingerprint`**: `sha256("{album_id}|{norm_variant}|{norm_retailer}")` scoped to the parent `album_id`.

Missing variants or retailers are assigned explicit `[standard]` tokens to prevent false-positive collapses with distinct pressings, `purchase_url` is omitted from `releases`, and duplicate submissions link to existing releases atomically via the `submission_releases` join table.
