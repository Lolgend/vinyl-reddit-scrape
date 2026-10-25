from datetime import datetime, timezone

import pytest
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.ingestion import (
    RawSubmission,
    RedditSourceAdapter,
    SubmissionIngester,
    should_ingest_submission,
)
from reddit_scrape.models import Submission, SubmissionSnapshot


@pytest.fixture(scope="session")
def engine():
    try:
        eng = get_engine()
        init_db(eng)
        return eng
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Database unavailable: {exc}")


@pytest.fixture(autouse=True)
def clean_db(engine):
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE submission_snapshots, submissions CASCADE;"))
    yield
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE submission_snapshots, submissions CASCADE;"))


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine)


def test_filter_non_release_flairs():
    """Verify that non-release flairs (Discussion, Question, Collection, Advice, ISO) are skipped."""
    for flair in ["Discussion", "Question", "Collection", "Advice", "ISO", "discussion", "iso"]:
        assert should_ingest_submission("Some Title", flair) is False

    for valid_flair in ["Pre-Order", "New Release", "Restock", "Repress", "Vinyl", None]:
        assert should_ingest_submission("Some Title", valid_flair) is True

    # Arbitrary keywords like PSA, warning, help are not filtered at ingestion stage
    for non_filtered_title in ["[PSA] Release Delayed", "[Warning] Bootleg alert", "Help wanted", "Survey on vinyl"]:
        assert should_ingest_submission(non_filtered_title, None) is True


def test_ingest_new_submission_persists_submission_and_snapshot(session_factory):
    """Verify idempotent insert sets is_extracted=False, is_release=True, and appends a snapshot."""
    ingester = SubmissionIngester(session_factory)
    created_dt = datetime.now(timezone.utc)

    raw = RawSubmission(
        id="sub101",
        title="Daft Punk - Discovery (2LP Gatefold)",
        url="https://example.com/daft-punk",
        created_utc=created_dt,
        score=42,
        upvote_ratio=0.95,
        num_comments=8,
        flair="Pre-Order",
    )

    result = ingester.ingest_submissions([raw])
    assert result.inserted == 1
    assert result.updated == 0
    assert result.skipped == 0

    with session_factory() as session:
        sub = session.execute(
            select(Submission).where(Submission.reddit_id == "sub101")
        ).scalar_one()

        assert sub.reddit_id == "sub101"
        assert sub.title == "Daft Punk - Discovery (2LP Gatefold)"
        assert sub.latest_score == 42
        assert sub.latest_upvote_ratio == 0.95
        assert sub.latest_num_comments == 8
        assert sub.flair == "Pre-Order"
        assert sub.is_extracted is False
        assert sub.is_release is True

        snapshots = session.execute(
            select(SubmissionSnapshot).where(SubmissionSnapshot.submission_id == sub.id)
        ).scalars().all()

        assert len(snapshots) == 1
        assert snapshots[0].score == 42
        assert snapshots[0].upvote_ratio == 0.95
        assert snapshots[0].num_comments == 8
        assert not hasattr(snapshots[0], "flair")
        assert snapshots[0].captured_at is not None


def test_ingest_duplicate_submission_upserts_and_appends_snapshot(session_factory):
    """Verify conflict on reddit_id updates metrics and unconditionally appends a new snapshot."""
    ingester = SubmissionIngester(session_factory)
    created_dt = datetime.now(timezone.utc)

    raw1 = RawSubmission(
        id="sub202",
        title="The Mars Volta - Frances the Mute (3LP)",
        url="https://example.com/tmv",
        created_utc=created_dt,
        score=10,
        upvote_ratio=0.88,
        num_comments=3,
        flair="New Release",
    )

    res1 = ingester.ingest_submissions([raw1])
    assert res1.inserted == 1
    assert res1.updated == 0

    # Observation cycle 2: updated metrics and flair mutation to SOLD OUT
    raw2 = RawSubmission(
        id="sub202",
        title="The Mars Volta - Frances the Mute (3LP)",
        url="https://example.com/tmv",
        created_utc=created_dt,
        score=180,
        upvote_ratio=0.97,
        num_comments=45,
        flair="SOLD OUT",
    )

    res2 = ingester.ingest_submissions([raw2])
    assert res2.inserted == 0
    assert res2.updated == 1

    with session_factory() as session:
        subs = session.execute(
            select(Submission).where(Submission.reddit_id == "sub202")
        ).scalars().all()
        assert len(subs) == 1
        sub = subs[0]
        assert sub.latest_score == 180
        assert sub.latest_upvote_ratio == 0.97
        assert sub.latest_num_comments == 45
        assert sub.flair == "SOLD OUT"

        snapshots = session.execute(
            select(SubmissionSnapshot)
            .where(SubmissionSnapshot.submission_id == sub.id)
            .order_by(SubmissionSnapshot.captured_at.asc())
        ).scalars().all()

        assert len(snapshots) == 2
        assert snapshots[0].score == 10
        assert not hasattr(snapshots[0], "flair")
        assert snapshots[1].score == 180
        assert not hasattr(snapshots[1], "flair")


def test_ingest_conflict_preserves_is_extracted_status(session_factory):
    """Verify that updating an already-extracted submission does not overwrite is_extracted to False."""
    ingester = SubmissionIngester(session_factory)
    created_dt = datetime.now(timezone.utc)

    raw1 = RawSubmission(
        id="sub303",
        title="Radiohead - Kid A Mnesia (3LP Red)",
        url="https://example.com/kid-a",
        created_utc=created_dt,
        score=50,
        upvote_ratio=0.92,
        num_comments=12,
        flair="Pre-Order",
    )

    ingester.ingest_submissions([raw1])

    with session_factory() as session:
        sub = session.execute(
            select(Submission).where(Submission.reddit_id == "sub303")
        ).scalar_one()
        sub.is_extracted = True
        session.commit()

    # Re-observe post with new metrics
    raw2 = RawSubmission(
        id="sub303",
        title="Radiohead - Kid A Mnesia (3LP Red)",
        url="https://example.com/kid-a",
        created_utc=created_dt,
        score=120,
        upvote_ratio=0.94,
        num_comments=25,
        flair="Pre-Order",
    )

    ingester.ingest_submissions([raw2])

    with session_factory() as session:
        sub = session.execute(
            select(Submission).where(Submission.reddit_id == "sub303")
        ).scalar_one()
        assert sub.latest_score == 120
        assert sub.is_extracted is True


def test_ingest_stream_skips_non_release_submissions(session_factory):
    """Verify batch ingestion skips non-release flairs while persisting valid releases."""
    ingester = SubmissionIngester(session_factory)
    created_dt = datetime.now(timezone.utc)

    items = [
        RawSubmission(
            id="sub_valid_1",
            title="Artist A - Album 1",
            url="https://example.com/1",
            created_utc=created_dt,
            score=15,
            flair="New Release",
        ),
        RawSubmission(
            id="sub_disc",
            title="[Discussion] What is your favorite album?",
            url="https://example.com/disc",
            created_utc=created_dt,
            score=50,
            flair="Discussion",
        ),
        RawSubmission(
            id="sub_iso",
            title="[ISO] Looking for Frank Ocean Blonde",
            url="https://example.com/iso",
            created_utc=created_dt,
            score=5,
            flair="ISO",
        ),
        RawSubmission(
            id="sub_valid_2",
            title="Artist B - Album 2",
            url="https://example.com/2",
            created_utc=created_dt,
            score=30,
            flair=None,
        ),
    ]

    result = ingester.ingest_submissions(items)
    assert result.inserted == 2
    assert result.skipped == 2

    with session_factory() as session:
        subs = session.execute(select(Submission)).scalars().all()
        reddit_ids = {s.reddit_id for s in subs}
        assert reddit_ids == {"sub_valid_1", "sub_valid_2"}


def test_reddit_source_adapter_converts_item():
    """Verify RedditSourceAdapter correctly normalizes submission items."""
    class MockPrawSubmission:
        id = "mock123"
        name = "t3_mock123"
        title = "Khruangbin - A La Sala"
        url = "https://example.com/khru"
        link_flair_text = "Pre-Order"
        score = 88
        upvote_ratio = 0.99
        num_comments = 14
        created_utc = 1714000000.0

    adapter = RedditSourceAdapter(client=None)
    raw = adapter.normalize_submission(MockPrawSubmission())

    assert raw.id == "mock123"
    assert raw.fullname == "t3_mock123"
    assert raw.title == "Khruangbin - A La Sala"
    assert raw.flair == "Pre-Order"
    assert raw.score == 88
    assert raw.created_utc == datetime.fromtimestamp(1714000000.0, tz=timezone.utc)


def test_reddit_source_adapter_fetch_methods():
    """Verify RedditSourceAdapter fetch_new and fetch_hot call client methods properly."""
    class FakeSubreddit:
        def __init__(self):
            self.new_called_with = None
            self.hot_called_with = None

        def new(self, limit):
            self.new_called_with = limit
            return [
                {
                    "id": "item_new",
                    "title": "Artist - New Album",
                    "url": "https://url.com/new",
                    "created_utc": 1714000000,
                }
            ]

        def hot(self, limit):
            self.hot_called_with = limit
            return [
                {
                    "id": "item_hot",
                    "title": "Artist - Hot Album",
                    "url": "https://url.com/hot",
                    "created_utc": 1714000000,
                }
            ]

    class FakeClient:
        def __init__(self):
            self.sub = FakeSubreddit()

        def subreddit(self, name):
            assert name == "vinylreleases"
            return self.sub

    fake_client = FakeClient()
    adapter = RedditSourceAdapter(client=fake_client)

    new_items = adapter.fetch_new(limit=25)
    assert fake_client.sub.new_called_with == 25
    assert len(new_items) == 1
    assert new_items[0].id == "item_new"

    hot_items = adapter.fetch_hot(limit=30)
    assert fake_client.sub.hot_called_with == 30
    assert len(hot_items) == 1
    assert hot_items[0].id == "item_hot"


def test_submission_schema_and_raw_submission_exclude_unused_metadata():
    """Verify that Submission model columns and RawSubmission fields omit permalink, author, is_self, selftext."""
    import dataclasses

    col_names = {c.name for c in Submission.__table__.columns}
    for obsolete_col in ("permalink", "author", "is_self", "selftext"):
        assert obsolete_col not in col_names, f"Column {obsolete_col} should not be in Submission schema"

    raw_fields = {f.name for f in dataclasses.fields(RawSubmission)}
    for obsolete_field in ("permalink", "author", "is_self", "selftext"):
        assert obsolete_field not in raw_fields, f"Field {obsolete_field} should not be in RawSubmission"


def test_submission_snapshot_schema_excludes_flair():
    """Verify that SubmissionSnapshot model columns omit flair and only define quantitative metrics."""
    col_names = {c.name for c in SubmissionSnapshot.__table__.columns}
    assert "flair" not in col_names, "Column flair should not be in SubmissionSnapshot schema"
    assert col_names == {"id", "submission_id", "score", "upvote_ratio", "num_comments", "captured_at"}
