# Deterministic Fingerprint for Release Identity and Deduplication

When multiple Reddit submissions announce the same vinyl release, deduplicating via retail URLs is unreliable because submissions frequently link to general storefronts, label hubs, or artist homepages rather than specific product SKUs. Furthermore, purchasing/procurement links are out of scope for the scraper pipeline.

We decided to uniquely identify releases in PostgreSQL using a deterministic SHA-256 `fingerprint` computed over normalized release attributes: `(combined_artist, album_title, variant, retailer)`. Missing variants or retailers are assigned explicit standard/unspecified tokens to prevent false-positive collapses with distinct pressings, `purchase_url` is removed from `releases`, and duplicate submissions link to existing releases atomically via the `submission_releases` join table.
