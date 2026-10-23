from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
from typing import Any, Iterable, List, Optional, Set

from sqlalchemy import func, literal_column
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from reddit_scrape.models import Submission, SubmissionSnapshot

NON_RELEASE_FLAIRS: Set[str] = {
    "discussion",
    "question",
    "collection",
    "advice",
    "iso",
}

NON_RELEASE_TITLE_PATTERN = re.compile(
    r"^\s*\[?(?:discussion|iso|question|psa|warning|help|survey|meta|advice|collection)\]?",
    re.IGNORECASE,
)


def should_ingest_submission(title: str, flair: Optional[str]) -> bool:
    """Check whether a submission should be ingested based on flair and title patterns."""
    if flair and flair.strip().lower() in NON_RELEASE_FLAIRS:
        return False

    if title and NON_RELEASE_TITLE_PATTERN.match(title.strip()):
        return False

    return True


@dataclass
class RawSubmission:
    """Canonical representation of an ingested raw Reddit submission item."""

    id: str  # Reddit base36 identifier (e.g. '123abc')
    title: str
    url: str
    permalink: str
    created_utc: datetime | float | int
    score: int = 0
    upvote_ratio: float = 1.0
    num_comments: int = 0
    flair: Optional[str] = None
    author: Optional[str] = None
    is_self: bool = False
    selftext: Optional[str] = None
    fullname: Optional[str] = None  # Reddit fullname (e.g. 't3_123abc')

    def get_created_utc_datetime(self) -> datetime:
        """Convert created_utc to a timezone-aware UTC datetime."""
        if isinstance(self.created_utc, (int, float)):
            return datetime.fromtimestamp(self.created_utc, tz=timezone.utc)
        elif self.created_utc.tzinfo is None:
            return self.created_utc.replace(tzinfo=timezone.utc)
        return self.created_utc


@dataclass
class IngestionStats:
    """Summary statistics for an ingestion batch."""

    inserted: int = 0
    updated: int = 0
    skipped: int = 0


class SubmissionIngester:
    """Service responsible for filtering, upserting submissions, and appending point-in-time snapshots."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def ingest_submission(self, item: RawSubmission, session: Session) -> Optional[str]:
        """Ingest a single submission item within an active database session.

        Returns 'inserted' if newly added, 'updated' if existing on conflict, or None if skipped.
        """
        if not should_ingest_submission(item.title, item.flair):
            return None

        sub_id = item.fullname or f"t3_{item.id}"
        created_dt = item.get_created_utc_datetime()

        stmt = pg_insert(Submission).values(
            id=sub_id,
            reddit_id=item.id,
            title=item.title,
            url=item.url,
            permalink=item.permalink,
            flair=item.flair,
            author=item.author,
            is_self=item.is_self,
            selftext=item.selftext,
            latest_score=item.score,
            latest_upvote_ratio=item.upvote_ratio,
            latest_num_comments=item.num_comments,
            created_utc=created_dt,
            is_extracted=False,
            is_release=True,
        )

        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=[Submission.reddit_id],
            set_={
                "flair": stmt.excluded.flair,
                "latest_score": stmt.excluded.latest_score,
                "latest_upvote_ratio": stmt.excluded.latest_upvote_ratio,
                "latest_num_comments": stmt.excluded.latest_num_comments,
                "updated_at": func.now(),
            },
        ).returning(Submission.id, literal_column("xmax = 0").label("is_inserted"))

        row = session.execute(upsert_stmt).fetchone()
        if not row:
            return None

        persisted_id, is_inserted = row[0], row[1]

        # Appending timestamped metric snapshot unconditionally on every observation cycle
        snapshot = SubmissionSnapshot(
            submission_id=persisted_id,
            score=item.score,
            upvote_ratio=item.upvote_ratio,
            num_comments=item.num_comments,
            flair=item.flair,
        )
        session.add(snapshot)

        return "inserted" if is_inserted else "updated"

    def ingest_submissions(self, items: Iterable[Any]) -> IngestionStats:
        """Ingest an iterable stream of submission items, committing atomically."""
        stats = IngestionStats()
        adapter = RedditSourceAdapter(client=None)

        with self.session_factory() as session:
            with session.begin():
                for item in items:
                    raw = adapter.normalize_submission(item)
                    status = self.ingest_submission(raw, session)
                    if status == "inserted":
                        stats.inserted += 1
                    elif status == "updated":
                        stats.updated += 1
                    else:
                        stats.skipped += 1

        return stats


class RedditSourceAdapter:
    """Adapter connecting to Reddit API via PRAW to ingest raw submissions."""

    def __init__(self, client: Optional[Any] = None) -> None:
        self.client = client

    def get_client(self) -> Any:
        """Lazily initialize PRAW Reddit instance from environment variables if not injected."""
        if self.client is None:
            import praw

            client_id = os.environ.get("REDDIT_CLIENT_ID")
            client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
            user_agent = os.environ.get(
                "REDDIT_USER_AGENT", "python:vinyl-releases-scraper:v0.1.0"
            )
            self.client = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
        return self.client

    def normalize_submission(self, item: Any) -> RawSubmission:
        """Normalize a PRAW Submission or dictionary into a canonical RawSubmission."""
        if isinstance(item, RawSubmission):
            return item

        def _to_dt(val: Any) -> datetime:
            if isinstance(val, (int, float)):
                return datetime.fromtimestamp(val, tz=timezone.utc)
            elif isinstance(val, datetime):
                return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
            raise ValueError(f"Invalid created_utc format: {val}")

        if isinstance(item, dict):
            return RawSubmission(
                id=item["id"],
                title=item["title"],
                url=item["url"],
                permalink=item["permalink"],
                created_utc=_to_dt(item["created_utc"]),
                score=item.get("score", 0),
                upvote_ratio=item.get("upvote_ratio", 1.0),
                num_comments=item.get("num_comments", 0),
                flair=item.get("flair") or item.get("link_flair_text"),
                author=item.get("author"),
                is_self=item.get("is_self", False),
                selftext=item.get("selftext"),
                fullname=item.get("fullname") or item.get("name"),
            )

        # Assuming PRAW Submission object
        author_name = None
        if hasattr(item, "author") and item.author:
            author_name = getattr(item.author, "name", str(item.author))

        return RawSubmission(
            id=item.id,
            title=item.title,
            url=item.url,
            permalink=item.permalink,
            created_utc=_to_dt(item.created_utc),
            score=getattr(item, "score", 0),
            upvote_ratio=getattr(item, "upvote_ratio", 1.0),
            num_comments=getattr(item, "num_comments", 0),
            flair=getattr(item, "link_flair_text", None) or getattr(item, "flair", None),
            author=author_name,
            is_self=getattr(item, "is_self", False),
            selftext=getattr(item, "selftext", None),
            fullname=getattr(item, "name", None) or getattr(item, "fullname", None),
        )

    def fetch_new(
        self, limit: int = 50, subreddit_name: str = "vinylreleases"
    ) -> List[RawSubmission]:
        """Fetch newest submissions from subreddit."""
        client = self.get_client()
        submissions = client.subreddit(subreddit_name).new(limit=limit)
        return [self.normalize_submission(s) for s in submissions]

    def fetch_hot(
        self, limit: int = 50, subreddit_name: str = "vinylreleases"
    ) -> List[RawSubmission]:
        """Fetch active/hot submissions from subreddit."""
        client = self.get_client()
        submissions = client.subreddit(subreddit_name).hot(limit=limit)
        return [self.normalize_submission(s) for s in submissions]
