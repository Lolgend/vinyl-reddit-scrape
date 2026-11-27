"""PostgreSQL integration tests for automatic PipelineRun retention in PollerService."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.models import (
    PipelineLoopName,
    PipelineLoopState,
    PipelineProcessStatus,
    PipelineRun,
    PipelineRunStatus,
    Submission,
    SubmissionSnapshot,
)
from reddit_scrape.service import PollerService
from reddit_scrape.telemetry import PipelineTelemetryRecorder


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


class _FakeClock:
    def __init__(self, initial: datetime) -> None:
        self.current = initial

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> datetime:
        self.current = self.current + timedelta(seconds=seconds)
        return self.current


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _run_exists(session_factory, run_id: int) -> bool:
    with session_factory() as session:
        return session.get(PipelineRun, run_id) is not None


class TestPollerServiceRunRetention:
    """Tests for automatic startup and 24-hour pipeline_runs pruning in PollerService.run()."""

    def test_run_prunes_expired_runs_on_startup_and_24h_cadence_preserving_snapshots(
        self, session_factory, monkeypatch
    ):
        monkeypatch.delenv("PIPELINE_RUN_RETENTION_DAYS", raising=False)
        monkeypatch.delenv("SNAPSHOT_RETENTION_DAYS", raising=False)

        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        with session_factory() as session:
            sub = Submission(
                id="sub_ret_adr0005",
                reddit_id="t3_adr0005",
                title="Boards of Canada - Music Has the Right to Children [2LP]",
                url="https://example.com/boc",
                latest_score=100,
                latest_upvote_ratio=0.98,
                latest_num_comments=15,
                created_utc=t0 - timedelta(days=45),
                is_extracted=True,
                is_release=True,
            )
            session.add(sub)
            session.flush()

            old_snapshot = SubmissionSnapshot(
                submission_id=sub.id,
                score=10,
                upvote_ratio=0.95,
                num_comments=2,
                captured_at=t0 - timedelta(days=45),
            )
            expired_run = PipelineRun(
                loop_name=PipelineLoopName.DISCOVERY_POLL,
                status=PipelineRunStatus.SUCCESS,
                started_at=t0 - timedelta(days=35),
                completed_at=t0 - timedelta(days=35) + timedelta(seconds=1),
                duration_ms=1000,
            )
            borderline_run = PipelineRun(
                loop_name=PipelineLoopName.TRACKING_POLL,
                status=PipelineRunStatus.SUCCESS,
                started_at=t0 - timedelta(days=29, hours=12),
                completed_at=t0 - timedelta(days=29, hours=12) + timedelta(seconds=1),
                duration_ms=1000,
            )
            fresh_run = PipelineRun(
                loop_name=PipelineLoopName.ENRICHMENT_WORKER,
                status=PipelineRunStatus.SUCCESS,
                started_at=t0 - timedelta(days=5),
                completed_at=t0 - timedelta(days=5) + timedelta(seconds=1),
                duration_ms=1000,
            )
            session.add_all([old_snapshot, expired_run, borderline_run, fresh_run])
            session.commit()
            old_snapshot_id = old_snapshot.id
            expired_run_id = expired_run.id
            borderline_run_id = borderline_run.id
            fresh_run_id = fresh_run.id

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            poll_interval_seconds=172800,
            hot_poll_interval_seconds=172800,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()
        try:
            # Wait for initial startup polls and startup purge to complete
            assert _wait_until(
                lambda: (
                    not _run_exists(session_factory, expired_run_id)
                    and _count_runs(session_factory) >= 4
                )
            )
            # Confirm borderline and fresh runs, process/loop state, and old snapshot remain intact
            with session_factory() as session:
                assert session.get(PipelineRun, expired_run_id) is None
                assert session.get(PipelineRun, borderline_run_id) is not None
                assert session.get(PipelineRun, fresh_run_id) is not None
                assert session.get(SubmissionSnapshot, old_snapshot_id) is not None
                assert session.get(PipelineProcessStatus, "default") is not None
                assert (
                    session.get(PipelineLoopState, PipelineLoopName.DISCOVERY_POLL)
                    is not None
                )

            # Advance clock to 23h -> 24h prune must NOT have fired yet
            clock.current = t0 + timedelta(hours=23)
            time.sleep(1.2)
            assert _run_exists(session_factory, borderline_run_id) is True

            # Advance clock to 24h (86400s) -> 24h prune fires and deletes borderline_run
            clock.current = t0 + timedelta(seconds=86400)
            assert _wait_until(
                lambda: not _run_exists(session_factory, borderline_run_id)
            )

            with session_factory() as session:
                assert session.get(PipelineRun, borderline_run_id) is None
                assert session.get(PipelineRun, fresh_run_id) is not None
                # ADR-0005: submission_snapshots remains strictly manual
                assert session.get(SubmissionSnapshot, old_snapshot_id) is not None
        finally:
            service.stop()
            run_thread.join(timeout=5.0)

    def test_run_uses_default_pipeline_run_retention_days_env_var(
        self, session_factory, monkeypatch
    ):
        monkeypatch.setenv("PIPELINE_RUN_RETENTION_DAYS", "7")

        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        with session_factory() as session:
            ten_days_old = PipelineRun(
                loop_name=PipelineLoopName.DISCOVERY_POLL,
                status=PipelineRunStatus.SUCCESS,
                started_at=t0 - timedelta(days=10),
                completed_at=t0 - timedelta(days=10) + timedelta(seconds=1),
                duration_ms=1000,
            )
            three_days_old = PipelineRun(
                loop_name=PipelineLoopName.TRACKING_POLL,
                status=PipelineRunStatus.SUCCESS,
                started_at=t0 - timedelta(days=3),
                completed_at=t0 - timedelta(days=3) + timedelta(seconds=1),
                duration_ms=1000,
            )
            session.add_all([ten_days_old, three_days_old])
            session.commit()
            ten_days_id = ten_days_old.id
            three_days_id = three_days_old.id

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            poll_interval_seconds=172800,
            hot_poll_interval_seconds=172800,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()
        try:
            assert _wait_until(lambda: not _run_exists(session_factory, ten_days_id))
            with session_factory() as session:
                assert session.get(PipelineRun, ten_days_id) is None
                assert session.get(PipelineRun, three_days_id) is not None
        finally:
            service.stop()
            run_thread.join(timeout=5.0)

    def test_run_defensively_logs_warning_when_purge_fails_without_crashing(
        self, session_factory, caplog
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            poll_interval_seconds=172800,
            hot_poll_interval_seconds=172800,
            telemetry_recorder=recorder,
            clock=clock,
        )

        with (
            caplog.at_level(logging.WARNING, logger="reddit_scrape.service"),
            patch(
                "reddit_scrape.service.purge_pipeline_runs",
                side_effect=RuntimeError("db lock timeout"),
            ) as mock_purge,
        ):
            run_thread = threading.Thread(target=service.run, daemon=True)
            run_thread.start()
            try:
                assert _wait_until(lambda: mock_purge.call_count >= 1)
                # Pipeline still ran its initial discovery & tracking polls
                assert _wait_until(lambda: _count_runs(session_factory) >= 2)

                # Advance 24 hours and verify second failure is also caught & logged
                clock.current = t0 + timedelta(seconds=86400)
                assert _wait_until(lambda: mock_purge.call_count >= 2)
            finally:
                service.stop()
                run_thread.join(timeout=5.0)

        warning_messages = [
            rec.message for rec in caplog.records if rec.levelno == logging.WARNING
        ]
        assert len(warning_messages) >= 2
        assert any("db lock timeout" in msg for msg in warning_messages)


def _count_runs(session_factory) -> int:
    with session_factory() as session:
        return len(session.scalars(select(PipelineRun)).all())
