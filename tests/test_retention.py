import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

from reddit_scrape.cli import main as cli_main
from reddit_scrape.db import get_engine, init_db
from reddit_scrape.models import (
    Album,
    AlbumArtist,
    Artist,
    PipelineLoopName,
    PipelineLoopState,
    PipelineLoopStatus,
    PipelineProcessState,
    PipelineProcessStatus,
    PipelineRun,
    PipelineRunMode,
    PipelineRunStatus,
    Release,
    Submission,
    SubmissionRelease,
    SubmissionSnapshot,
)

# Seams under test - will fail to import until implemented
from reddit_scrape.retention import purge_snapshots


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
        conn.execute(
            text(
                "TRUNCATE TABLE pipeline_runs, pipeline_loop_states, pipeline_process_status, "
                "submission_snapshots, submission_releases, releases, album_artists, albums, "
                "artists, submissions CASCADE;"
            )
        )
    yield
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE pipeline_runs, pipeline_loop_states, pipeline_process_status, "
                "submission_snapshots, submission_releases, releases, album_artists, albums, "
                "artists, submissions CASCADE;"
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

        # Create associated artist, album, release, and join records
        artist = Artist(name="Daft Punk")
        album = Album(
            combined_artist="Daft Punk",
            title="Discovery",
            fingerprint="fake_album_fp_retention_01",
        )
        session.add_all([artist, album])
        session.flush()

        release = Release(
            album_id=album.id,
            variant="Gatefold 2LP",
            retailer_name="Standard",
            fingerprint="fake_fingerprint_retention_01",
        )
        session.add(release)
        session.flush()

        alb_artist = AlbumArtist(album_id=album.id, artist_id=artist.id)
        sub_rel = SubmissionRelease(submission_id=sub.id, release_id=release.id)
        session.add_all([alb_artist, sub_rel])

        # Add snapshots: 2 older than 30 days, 2 within 30 days
        snap_old1 = SubmissionSnapshot(
            submission_id=sub.id,
            score=10,
            upvote_ratio=1.0,
            num_comments=2,
            captured_at=now - timedelta(days=45),
        )
        snap_old2 = SubmissionSnapshot(
            submission_id=sub.id,
            score=50,
            upvote_ratio=0.95,
            num_comments=12,
            captured_at=now - timedelta(days=35),
        )
        snap_recent1 = SubmissionSnapshot(
            submission_id=sub.id,
            score=180,
            upvote_ratio=0.97,
            num_comments=30,
            captured_at=now - timedelta(days=15),
        )
        snap_recent2 = SubmissionSnapshot(
            submission_id=sub.id,
            score=250,
            upvote_ratio=0.98,
            num_comments=45,
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
        alb_count = session.execute(select(Album)).scalars().all()
        assert len(alb_count) == 1
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
        check=False,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "purged" in result.stdout.lower()


def test_get_default_pipeline_run_retention_days_precedence_and_export(monkeypatch):
    """Verify env fallback precedence (PIPELINE_RUN_RETENTION_DAYS -> SNAPSHOT_RETENTION_DAYS -> 30)
    with invalid non-integer values ignored, and top-level package export.
    """
    import reddit_scrape
    from reddit_scrape.retention import get_default_pipeline_run_retention_days

    assert reddit_scrape.get_default_pipeline_run_retention_days is get_default_pipeline_run_retention_days
    assert "get_default_pipeline_run_retention_days" in reddit_scrape.__all__

    monkeypatch.delenv("PIPELINE_RUN_RETENTION_DAYS", raising=False)
    monkeypatch.delenv("SNAPSHOT_RETENTION_DAYS", raising=False)
    assert get_default_pipeline_run_retention_days() == 30

    monkeypatch.setenv("SNAPSHOT_RETENTION_DAYS", "14")
    assert get_default_pipeline_run_retention_days() == 14

    monkeypatch.setenv("PIPELINE_RUN_RETENTION_DAYS", "7")
    assert get_default_pipeline_run_retention_days() == 7

    # Invalid PIPELINE_RUN_RETENTION_DAYS falls back cleanly to SNAPSHOT_RETENTION_DAYS
    monkeypatch.setenv("PIPELINE_RUN_RETENTION_DAYS", "not-an-int")
    assert get_default_pipeline_run_retention_days() == 14

    # Both invalid fall back cleanly to 30
    monkeypatch.setenv("SNAPSHOT_RETENTION_DAYS", "invalid")
    assert get_default_pipeline_run_retention_days() == 30


def test_purge_pipeline_runs_deletes_old_preserves_recent_and_state(session_factory):
    """Verify purge_pipeline_runs deletes expired runs, preserves newer runs and
    operational state tables, and is exported from top-level reddit_scrape.
    """
    import reddit_scrape
    from reddit_scrape.retention import purge_pipeline_runs

    assert reddit_scrape.purge_pipeline_runs is purge_pipeline_runs
    assert "purge_pipeline_runs" in reddit_scrape.__all__

    now = datetime.now(timezone.utc)

    with session_factory() as session:
        session.add(
            PipelineProcessStatus(
                process_name="default",
                status=PipelineProcessState.RUNNING,
                run_mode=PipelineRunMode.CONTINUOUS,
                host="worker-1",
                pid=1234,
                heartbeat_interval_seconds=30,
                started_at=now - timedelta(days=45),
                last_heartbeat_at=now,
            )
        )
        session.add(
            PipelineLoopState(
                loop_name=PipelineLoopName.DISCOVERY_POLL,
                status=PipelineLoopStatus.SLEEPING,
                interval_seconds=300,
                last_started_at=now - timedelta(minutes=5),
                last_completed_at=now - timedelta(minutes=5),
                next_run_at=now + timedelta(minutes=5),
            )
        )
        session.add_all(
            [
                PipelineRun(
                    loop_name=PipelineLoopName.DISCOVERY_POLL,
                    status=PipelineRunStatus.SUCCESS,
                    started_at=now - timedelta(days=45),
                    completed_at=now - timedelta(days=45) + timedelta(seconds=2),
                    duration_ms=2000,
                ),
                PipelineRun(
                    loop_name=PipelineLoopName.TRACKING_POLL,
                    status=PipelineRunStatus.ERROR,
                    started_at=now - timedelta(days=31),
                    completed_at=now - timedelta(days=31) + timedelta(seconds=1),
                    duration_ms=1000,
                    error_message="temporary failure",
                ),
                PipelineRun(
                    loop_name=PipelineLoopName.DISCOVERY_POLL,
                    status=PipelineRunStatus.SUCCESS,
                    started_at=now - timedelta(days=29),
                    completed_at=now - timedelta(days=29) + timedelta(seconds=1),
                    duration_ms=1000,
                ),
                PipelineRun(
                    loop_name=PipelineLoopName.ENRICHMENT_WORKER,
                    status=PipelineRunStatus.SUCCESS,
                    started_at=now - timedelta(hours=2),
                    completed_at=now - timedelta(hours=2) + timedelta(seconds=3),
                    duration_ms=3000,
                ),
            ]
        )
        session.commit()

    deleted = purge_pipeline_runs(session_factory, retention_days=30, batch_size=100)
    assert deleted == 2

    with session_factory() as session:
        remaining = session.scalars(
            select(PipelineRun).order_by(PipelineRun.started_at.asc())
        ).all()
        assert len(remaining) == 2
        assert remaining[0].started_at == now - timedelta(days=29)
        assert remaining[1].started_at == now - timedelta(hours=2)

        proc = session.get(PipelineProcessStatus, "default")
        assert proc is not None
        assert proc.status == PipelineProcessState.RUNNING

        loop_state = session.get(PipelineLoopState, PipelineLoopName.DISCOVERY_POLL)
        assert loop_state is not None
        assert loop_state.status == PipelineLoopStatus.SLEEPING


def test_purge_pipeline_runs_chunked_batches_and_explicit_cutoff(session_factory):
    """Verify multi-batch deletion ordered across batches, direct Session support, and cutoff override."""
    from reddit_scrape.retention import purge_pipeline_runs

    t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    cutoff = t0 - timedelta(days=30)

    with session_factory() as session:
        for i in range(5):
            session.add(
                PipelineRun(
                    loop_name=PipelineLoopName.DISCOVERY_POLL,
                    status=PipelineRunStatus.SUCCESS,
                    started_at=cutoff - timedelta(days=1, hours=i),
                    completed_at=cutoff - timedelta(days=1, hours=i) + timedelta(seconds=1),
                    duration_ms=1000,
                )
            )
        # Exactly at cutoff -> strictly `< cutoff` means NOT deleted
        session.add(
            PipelineRun(
                loop_name=PipelineLoopName.DISCOVERY_POLL,
                status=PipelineRunStatus.SUCCESS,
                started_at=cutoff,
                completed_at=cutoff + timedelta(seconds=1),
                duration_ms=1000,
            )
        )
        session.commit()

        deleted = purge_pipeline_runs(
            session,
            retention_days=30,
            batch_size=2,
            cutoff=cutoff,
        )
        assert deleted == 5

    with session_factory() as session:
        remaining = session.scalars(select(PipelineRun)).all()
        assert len(remaining) == 1
        assert remaining[0].started_at == cutoff


def test_purge_pipeline_runs_validation(session_factory):
    """Verify input validation for invalid retention_days or batch_size."""
    from reddit_scrape.retention import purge_pipeline_runs

    with pytest.raises(ValueError, match="retention_days"):
        purge_pipeline_runs(session_factory, retention_days=-1)

    with pytest.raises(ValueError, match="batch_size"):
        purge_pipeline_runs(session_factory, retention_days=30, batch_size=0)

    with pytest.raises(ValueError, match="batch_size"):
        purge_pipeline_runs(session_factory, retention_days=30, batch_size=-5)


def test_cli_purge_pipeline_runs(session_factory, capsys):
    """Verify CLI interface for purge-pipeline-runs subcommand with PostgreSQL."""
    now = datetime.now(timezone.utc)

    with session_factory() as session:
        session.add_all(
            [
                PipelineRun(
                    loop_name=PipelineLoopName.DISCOVERY_POLL,
                    status=PipelineRunStatus.SUCCESS,
                    started_at=now - timedelta(days=35),
                    completed_at=now - timedelta(days=35) + timedelta(seconds=1),
                    duration_ms=1000,
                ),
                PipelineRun(
                    loop_name=PipelineLoopName.TRACKING_POLL,
                    status=PipelineRunStatus.SUCCESS,
                    started_at=now - timedelta(days=5),
                    completed_at=now - timedelta(days=5) + timedelta(seconds=1),
                    duration_ms=1000,
                ),
            ]
        )
        session.commit()

    exit_code = cli_main(["purge-pipeline-runs", "--days", "30", "--batch-size", "10"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Successfully purged 1 pipeline runs older than 30 days." in captured.out

    with session_factory() as session:
        remaining = session.scalars(select(PipelineRun)).all()
        assert len(remaining) == 1
        assert remaining[0].loop_name == PipelineLoopName.TRACKING_POLL


def test_cli_purge_pipeline_runs_env_var(session_factory, monkeypatch, capsys):
    """Verify purge-pipeline-runs CLI defaults to PIPELINE_RUN_RETENTION_DAYS when --days is omitted."""
    now = datetime.now(timezone.utc)

    with session_factory() as session:
        session.add_all(
            [
                PipelineRun(
                    loop_name=PipelineLoopName.DISCOVERY_POLL,
                    status=PipelineRunStatus.SUCCESS,
                    started_at=now - timedelta(days=20),
                    completed_at=now - timedelta(days=20) + timedelta(seconds=1),
                    duration_ms=1000,
                ),
                PipelineRun(
                    loop_name=PipelineLoopName.TRACKING_POLL,
                    status=PipelineRunStatus.SUCCESS,
                    started_at=now - timedelta(days=5),
                    completed_at=now - timedelta(days=5) + timedelta(seconds=1),
                    duration_ms=1000,
                ),
            ]
        )
        session.commit()

    monkeypatch.setenv("PIPELINE_RUN_RETENTION_DAYS", "15")

    exit_code = cli_main(["purge-pipeline-runs"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Successfully purged 1 pipeline runs older than 15 days." in captured.out

    with session_factory() as session:
        remaining = session.scalars(select(PipelineRun)).all()
        assert len(remaining) == 1
        assert remaining[0].loop_name == PipelineLoopName.TRACKING_POLL


def test_python_m_subprocess_purge_pipeline_runs():
    """Verify CLI runnable via python -m reddit_scrape purge-pipeline-runs --days 30."""
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = src_dir
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", "reddit_scrape", "purge-pipeline-runs", "--days", "30"],
        capture_output=True,
        check=False,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "pipeline runs" in result.stdout.lower()
