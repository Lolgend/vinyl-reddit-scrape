# Decouple Submissions and Releases

Reddit submissions from `r/vinylreleases` contain unstructured text, varied flairs, and occasional multi-album announcements. We decided to store raw Reddit submissions and extracted vinyl releases in separate relational tables linked by a many-to-many relationship (`submission_releases`). This ensures scraping is idempotent, raw post history is preserved, and extraction prompts can be updated and re-run over historical submissions without data loss.
