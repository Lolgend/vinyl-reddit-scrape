"""Shared test fixtures, clocks, polling helpers, and model factories."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest
from dotenv import load_dotenv
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.ingestion import RawSubmission
from reddit_scrape.models import (
    PipelineLoopName,
    PipelineLoopState,
    Submission,
)

DEFAULT_TIME = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

ALL_TABLES_TRUNCATE_SQL = (
    "TRUNCATE TABLE pipeline_runs, pipeline_loop_states, pipeline_process_status, "
    "submission_snapshots, submission_releases, releases, album_artists, albums, "
    "artists, submissions CASCADE;"
)


@pytest.fixture(scope="session")
def engine() -> Engine:
    """Create and initialize a session-scoped PostgreSQL test engine."""
    try:
        eng = get_engine()
        init_db(eng)
        return eng
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Database unavailable: {exc}")


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a SQLAlchemy sessionmaker bound to the test engine."""
    return sessionmaker(bind=engine)


def truncate_all_tables(engine: Engine) -> None:
    """Truncate all 10 application and telemetry tables."""
    with engine.begin() as conn:
        conn.execute(text(ALL_TABLES_TRUNCATE_SQL))


class FakeClock:
    """Deterministic mutable UTC clock for pipeline telemetry tests."""

    def __init__(self, initial: datetime = DEFAULT_TIME) -> None:
        self.current = initial

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> datetime:
        self.current = self.current + timedelta(seconds=seconds)
        return self.current


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = 5.0,
    interval: float = 0.02,
) -> bool:
    """Poll predicate until it returns True or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def get_loop_state(
    session_factory: sessionmaker[Session],
    loop_name: PipelineLoopName,
) -> PipelineLoopState | None:
    """Fetch a PipelineLoopState row by loop_name in an isolated read session."""
    with session_factory() as session:
        return session.get(PipelineLoopState, loop_name)


def make_raw_submission(
    sub_id: str = "sub_1",
    *,
    title: str = "Artist - Album [LP]",
    url: str | None = None,
    created_utc: datetime | float | None = None,
    score: int = 0,
    upvote_ratio: float = 1.0,
    num_comments: int = 0,
    flair: str | None = None,
    fullname: str | None = None,
) -> RawSubmission:
    """Build a RawSubmission dataclass with sensible defaults for tests."""
    return RawSubmission(
        id=sub_id,
        title=title,
        url=url or f"https://reddit.com/r/vinylreleases/{sub_id}",
        created_utc=created_utc if created_utc is not None else DEFAULT_TIME,
        score=score,
        upvote_ratio=upvote_ratio,
        num_comments=num_comments,
        flair=flair,
        fullname=fullname if fullname is not None else f"t3_{sub_id}",
    )


def make_submission(
    sub_id: str = "t3_sub_1",
    *,
    reddit_id: str | None = None,
    title: str = "Artist - Album [LP]",
    url: str | None = None,
    flair: str | None = None,
    latest_score: int = 0,
    latest_upvote_ratio: float = 1.0,
    latest_num_comments: int = 0,
    created_utc: datetime | None = None,
    is_extracted: bool = False,
    is_release: bool = True,
) -> Submission:
    """Build an ORM Submission model with sensible defaults for tests."""
    resolved_reddit_id = reddit_id or sub_id.removeprefix("t3_")
    return Submission(
        id=sub_id,
        reddit_id=resolved_reddit_id,
        title=title,
        url=url or f"https://reddit.com/r/vinylreleases/{resolved_reddit_id}",
        flair=flair,
        latest_score=latest_score,
        latest_upvote_ratio=latest_upvote_ratio,
        latest_num_comments=latest_num_comments,
        created_utc=created_utc if created_utc is not None else DEFAULT_TIME,
        is_extracted=is_extracted,
        is_release=is_release,
    )
