import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.models import (
    Artist,
    Release,
    ReleaseArtist,
    Submission,
    SubmissionRelease,
    SubmissionSnapshot,
)

# Seams under test - will fail to import until implemented
from reddit_scrape.retention import purge_snapshots
from reddit_scrape.cli import main as cli_main


@pytest.fixture(scope="session")
def engine():
    try:
        eng = get_engine()
        init_db(eng)
        return eng
    except Exception as exc:
        pytest.skip(f"Database unavailable: {exc}")


@pytest.fixture(autouse=True)
def clean_db(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE submission_releases, release_artists, releases, artists, submission_snapshots, submissions CASCADE;"
            )
        )
    yield
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE submission_releases, release_artists, releases, artists, submission_snapshots, submissions CASCADE;"
            )
        )


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine)


def test_purge_old_snapshots_keeps_recent(session_factory):
    """Verify that snapshots older than retention_days are purged while recent ones are kept,

    and parent submissions, releases, and artists remain completely intact with terminal metrics.
    """
    now = datetime.now(timezone.utc)

    with session_factory() as session:
        # Create parent submission with terminal metrics
        sub = Submission(
            id="sub_test_01",
            reddit_id="t3_retention1",
            title="Daft Punk - Discovery (Gatefold 2LP)",
            url="https://example.com/daft-punk",
            flair="Sold Out",
            latest_score=250,
            latest_upvote_ratio=0.98,
            latest_num_comments=45,
            created_utc=now - timedelta(days=60),
            is_extracted=True,
            is_release=True,
        )
        session.add(sub)

        # Create associated artist, release, and join records
        artist = Artist(name="Daft Punk")
        release = Release(
            combined_artist="Daft Punk",
            album_title="Discovery",
            variant="Gatefold 2LP",
            retailer_name="Standard",
            fingerprint="fake_fingerprint_retention_01",
        )
        session.add_all([artist, release])
        session.flush()

        rel_artist = ReleaseArtist(release_id=release.id, artist_id=artist.id)
        sub_rel = SubmissionRelease(submission_id=sub.id, release_id=release.id)
        session.add_all([rel_artist, sub_rel])

        # Add snapshots: 2 older than 30 days, 2 within 30 days
        snap_old1 = SubmissionSnapshot(
            submission_id=sub.id,
            score=10,
            upvote_ratio=1.0,
            num_comments=2,
            flair="New Release",
            captured_at=now - timedelta(days=45),
        )
        snap_old2 = SubmissionSnapshot(
            submission_id=sub.id,
            score=50,
            upvote_ratio=0.95,
            num_comments=12,
            flair="New Release",
            captured_at=now - timedelta(days=35),
        )
        snap_recent1 = SubmissionSnapshot(
            submission_id=sub.id,
            score=180,
            upvote_ratio=0.97,
            num_comments=30,
            flair="Pre-Order",
            captured_at=now - timedelta(days=15),
        )
        snap_recent2 = SubmissionSnapshot(
            submission_id=sub.id,
            score=250,
            upvote_ratio=0.98,
            num_comments=45,
            flair="Sold Out",
            captured_at=now - timedelta(days=2),
        )
        session.add_all([snap_old1, snap_old2, snap_recent1, snap_recent2])
        session.commit()

    # Execute purge with 30-day retention
    deleted_count = purge_snapshots(session_factory, retention_days=30, batch_size=100)
    assert deleted_count == 2

    with session_factory() as session:
        # Check remaining snapshots
        remaining_snaps = session.execute(
            select(SubmissionSnapshot).order_by(SubmissionSnapshot.captured_at.asc())
        ).scalars().all()
        assert len(remaining_snaps) == 2
        assert remaining_snaps[0].score == 180
        assert remaining_snaps[1].score == 250

        # Check parent submission is completely intact and preserves terminal metrics
        sub_db = session.execute(
            select(Submission).where(Submission.id == "sub_test_01")
        ).scalar_one()
        assert sub_db.latest_score == 250
        assert sub_db.latest_upvote_ratio == 0.98
        assert sub_db.latest_num_comments == 45
        assert sub_db.flair == "Sold Out"
        assert sub_db.is_extracted is True

        # Check related entities remain intact
        art_count = session.execute(select(Artist)).scalars().all()
        assert len(art_count) == 1
        rel_count = session.execute(select(Release)).scalars().all()
        assert len(rel_count) == 1
        sub_rel_count = session.execute(select(SubmissionRelease)).scalars().all()
        assert len(sub_rel_count) == 1


def test_purge_chunked_batched_deletion(session_factory):
    """Verify that batched chunked deletion deletes all eligible rows across multiple batches

    when total eligible rows exceeds batch_size (e.g. 5 snapshots deleted in batches of 2).
    """
    now = datetime.now(timezone.utc)

    with session_factory() as session:
        sub = Submission(
            id="sub_test_02",
            reddit_id="t3_retention2",
            title="Radiohead - OK Computer",
            url="https://example.com/radiohead",
            latest_score=300,
            latest_upvote_ratio=0.99,
            latest_num_comments=80,
            created_utc=now - timedelta(days=50),
            is_extracted=False,
            is_release=True,
        )
        session.add(sub)
        session.flush()

        # Insert 5 snapshots older than 10 days
        for i in range(5):
            snap = SubmissionSnapshot(
                submission_id=sub.id,
                score=10 * (i + 1),
                upvote_ratio=1.0,
                num_comments=i,
                captured_at=now - timedelta(days=20 + i),
            )
            session.add(snap)
        session.commit()

    # Purge with batch_size=2 and retention_days=10 -> should take 3 iterations (2, 2, 1) + 1 termination (0)
    deleted_count = purge_snapshots(session_factory, retention_days=10, batch_size=2)
    assert deleted_count == 5

    with session_factory() as session:
        snaps = session.execute(select(SubmissionSnapshot)).scalars().all()
        assert len(snaps) == 0

        # Parent submission still exists with terminal metrics preserved even when all snapshots purged
        sub_db = session.execute(
            select(Submission).where(Submission.id == "sub_test_02")
        ).scalar_one()
        assert sub_db.latest_score == 300
        assert sub_db.latest_upvote_ratio == 0.99
        assert sub_db.latest_num_comments == 80


def test_purge_when_no_snapshots_eligible(session_factory):
    """Verify that purge returns 0 and leaves data untouched when no snapshots are older than threshold."""
    now = datetime.now(timezone.utc)

    with session_factory() as session:
        sub = Submission(
            id="sub_test_03",
            reddit_id="t3_retention3",
            title="The Beatles - Abbey Road",
            url="https://example.com/beatles",
            latest_score=100,
            latest_upvote_ratio=0.95,
            latest_num_comments=10,
            created_utc=now - timedelta(days=5),
            is_extracted=False,
            is_release=True,
        )
        session.add(sub)
        session.flush()

        snap = SubmissionSnapshot(
            submission_id=sub.id,
            score=100,
            upvote_ratio=0.95,
            num_comments=10,
            captured_at=now - timedelta(days=2),
        )
        session.add(snap)
        session.commit()

    deleted_count = purge_snapshots(session_factory, retention_days=30, batch_size=100)
    assert deleted_count == 0

    with session_factory() as session:
        snaps = session.execute(select(SubmissionSnapshot)).scalars().all()
        assert len(snaps) == 1


def test_purge_snapshots_validation(session_factory):
    """Verify input validation for invalid retention days or batch sizes."""
    with pytest.raises(ValueError, match="retention_days"):
        purge_snapshots(session_factory, retention_days=-1)

    with pytest.raises(ValueError, match="batch_size"):
        purge_snapshots(session_factory, retention_days=30, batch_size=0)

    with pytest.raises(ValueError, match="batch_size"):
        purge_snapshots(session_factory, retention_days=30, batch_size=-5)


def test_cli_purge_snapshots(session_factory, monkeypatch, capsys):
    """Verify CLI interface for purge-snapshots subcommands with args and env vars."""
    now = datetime.now(timezone.utc)

    # Insert test data
    with session_factory() as session:
        sub = Submission(
            id="sub_cli_01",
            reddit_id="t3_cli1",
            title="Pink Floyd - The Dark Side of the Moon",
            url="https://example.com/pink-floyd",
            latest_score=500,
            latest_upvote_ratio=0.99,
            latest_num_comments=120,
            created_utc=now - timedelta(days=40),
            is_extracted=False,
            is_release=True,
        )
        session.add(sub)
        session.flush()

        snap1 = SubmissionSnapshot(
            submission_id=sub.id,
            score=50,
            upvote_ratio=0.9,
            num_comments=10,
            captured_at=now - timedelta(days=35),
        )
        snap2 = SubmissionSnapshot(
            submission_id=sub.id,
            score=500,
            upvote_ratio=0.99,
            num_comments=120,
            captured_at=now - timedelta(days=5),
        )
        session.add_all([snap1, snap2])
        session.commit()

    # Call CLI entrypoint with --days 30 --batch-size 10
    exit_code = cli_main(["purge-snapshots", "--days", "30", "--batch-size", "10"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "1" in captured.out or "purged" in captured.out.lower()

    with session_factory() as session:
        snaps = session.execute(select(SubmissionSnapshot)).scalars().all()
        assert len(snaps) == 1
        assert snaps[0].score == 500


def test_cli_default_env_var(session_factory, monkeypatch, capsys):
    """Verify CLI defaults to SNAPSHOT_RETENTION_DAYS environment variable when --days is omitted."""
    now = datetime.now(timezone.utc)

    with session_factory() as session:
        sub = Submission(
            id="sub_cli_env",
            reddit_id="t3_clienv",
            title="Fleetwood Mac - Rumours",
            url="https://example.com/fleetwood",
            latest_score=150,
            latest_upvote_ratio=0.96,
            latest_num_comments=30,
            created_utc=now - timedelta(days=25),
            is_extracted=False,
            is_release=True,
        )
        session.add(sub)
        session.flush()

        snap = SubmissionSnapshot(
            submission_id=sub.id,
            score=50,
            upvote_ratio=0.9,
            num_comments=10,
            captured_at=now - timedelta(days=20),
        )
        session.add(snap)
        session.commit()

    # Set env var SNAPSHOT_RETENTION_DAYS to 15 -> snap (20 days old) should be purged
    monkeypatch.setenv("SNAPSHOT_RETENTION_DAYS", "15")

    exit_code = cli_main(["purge-snapshots"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "purged" in captured.out.lower()

    with session_factory() as session:
        snaps = session.execute(select(SubmissionSnapshot)).scalars().all()
        assert len(snaps) == 0


def test_python_m_subprocess_execution():
    """Verify CLI runnable via python -m reddit_scrape purge-snapshots --days 30."""
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src_dir
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", "reddit_scrape", "purge-snapshots", "--days", "30"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "purged" in result.stdout.lower()
