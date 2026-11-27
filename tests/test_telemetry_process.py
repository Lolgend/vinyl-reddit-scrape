"""PostgreSQL integration and unit tests for Pipeline process liveness and heartbeat telemetry."""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

load_dotenv()

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.ingestion import RawSubmission
from reddit_scrape.models import (
    PipelineProcessState,
    PipelineProcessStatus,
    PipelineRunMode,
    Submission,
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
def clean_db(request):
    if request.cls and request.cls.__name__ == "TestPipelineTelemetryUnit":
        yield
        return
    engine = request.getfixturevalue("engine")
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE pipeline_process_status, submission_snapshots, "
                "submission_releases, releases, album_artists, albums, artists, submissions CASCADE;"
            )
        )
    yield
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE pipeline_process_status, submission_snapshots, "
                "submission_releases, releases, album_artists, albums, artists, submissions CASCADE;"
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


class TestPipelineTelemetryRecorder:
    """Tests for PipelineTelemetryRecorder PostgreSQL upserts and failure isolation."""

    def test_record_process_start_inserts_and_upserts_row(self, session_factory):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory,
            clock=clock,
            host="worker-host-1",
            pid=4242,
        )

        recorder.record_process_start(
            run_mode=PipelineRunMode.CONTINUOUS,
            heartbeat_interval_seconds=30,
        )

        with session_factory() as session:
            row = session.get(PipelineProcessStatus, "default")
            assert row is not None
            assert row.status == PipelineProcessState.RUNNING
            assert row.run_mode == PipelineRunMode.CONTINUOUS
            assert row.host == "worker-host-1"
            assert row.pid == 4242
            assert row.heartbeat_interval_seconds == 30
            assert row.started_at == t0
            assert row.last_heartbeat_at == t0
            assert row.stopped_at is None

        # Stop then restart with a new host/pid/mode and confirm upsert resets stopped_at
        t1 = clock.advance(60)
        recorder.record_process_stop()

        t2 = clock.advance(30)
        recorder_two = PipelineTelemetryRecorder(
            session_factory,
            clock=clock,
            host="worker-host-2",
            pid=9001,
        )
        recorder_two.record_process_start(
            run_mode=PipelineRunMode.ONCE,
            heartbeat_interval_seconds=15,
        )

        with session_factory() as session:
            row = session.get(PipelineProcessStatus, "default")
            assert row is not None
            assert row.status == PipelineProcessState.RUNNING
            assert row.run_mode == PipelineRunMode.ONCE
            assert row.host == "worker-host-2"
            assert row.pid == 9001
            assert row.heartbeat_interval_seconds == 15
            assert row.started_at == t2
            assert row.last_heartbeat_at == t2
            assert row.stopped_at is None
            assert t1 != t2

    def test_record_heartbeat_updates_last_heartbeat_preserving_metadata(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory,
            clock=clock,
            host="hb-host",
            pid=1111,
        )

        recorder.record_process_start(
            run_mode=PipelineRunMode.CONTINUOUS,
            heartbeat_interval_seconds=45,
        )

        t1 = clock.advance(45)
        recorder.record_heartbeat()

        with session_factory() as session:
            row = session.get(PipelineProcessStatus, "default")
            assert row is not None
            assert row.status == PipelineProcessState.RUNNING
            assert row.run_mode == PipelineRunMode.CONTINUOUS
            assert row.host == "hb-host"
            assert row.pid == 1111
            assert row.heartbeat_interval_seconds == 45
            assert row.started_at == t0
            assert row.last_heartbeat_at == t1
            assert row.stopped_at is None

    def test_record_process_stop_marks_stopped_and_preserves_started_at(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory,
            clock=clock,
            host="stop-host",
            pid=2222,
        )

        recorder.record_process_start(
            run_mode=PipelineRunMode.ONCE,
            heartbeat_interval_seconds=20,
        )

        t1 = clock.advance(12)
        recorder.record_process_stop()

        with session_factory() as session:
            row = session.get(PipelineProcessStatus, "default")
            assert row is not None
            assert row.status == PipelineProcessState.STOPPED
            assert row.run_mode == PipelineRunMode.ONCE
            assert row.host == "stop-host"
            assert row.pid == 2222
            assert row.heartbeat_interval_seconds == 20
            assert row.started_at == t0
            assert row.last_heartbeat_at == t1
            assert row.stopped_at == t1

class TestPipelineTelemetryUnit:
    """Unit tests for telemetry failure isolation without live DB."""

    def test_recorder_catches_db_errors_and_logs_warning(self, caplog):
        failing_factory = MagicMock(side_effect=RuntimeError("db connection lost"))
        recorder = PipelineTelemetryRecorder(
            failing_factory,
            clock=lambda: datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            host="err-host",
            pid=3333,
        )

        with caplog.at_level(logging.WARNING, logger="reddit_scrape.telemetry"):
            recorder.record_process_start(
                run_mode=PipelineRunMode.CONTINUOUS,
                heartbeat_interval_seconds=30,
            )
            recorder.record_heartbeat()
            recorder.record_process_stop()

        assert len(caplog.records) == 3
        assert all(r.levelno == logging.WARNING for r in caplog.records)


class TestPollerServiceProcessTelemetryIntegration:
    """PostgreSQL integration tests for PollerService process liveness and heartbeat."""

    def test_poller_service_default_telemetry_recorder_and_clock(self, session_factory):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)

        service = PollerService(
            session_factory=session_factory,
            heartbeat_interval_seconds=20,
            clock=clock,
        )
        try:
            assert isinstance(service.telemetry_recorder, PipelineTelemetryRecorder)
            assert service.heartbeat_interval_seconds == 20
            assert service._now() == t0
        finally:
            service.stop()

    def test_poller_service_run_records_start_heartbeat_and_graceful_stop(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory,
            clock=clock,
            host="run-host",
            pid=5555,
        )

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            poll_interval_seconds=60,
            hot_poll_interval_seconds=60,
            heartbeat_interval_seconds=1,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()

        try:
            # Wait for startup upsert and PipelineHeartbeat daemon thread
            deadline = time.monotonic() + 3.0
            heartbeat_thread = None
            row = None
            while time.monotonic() < deadline:
                with session_factory() as session:
                    row = session.get(PipelineProcessStatus, "default")
                for thread in threading.enumerate():
                    if thread.name == "PipelineHeartbeat":
                        heartbeat_thread = thread
                        break
                if (
                    row is not None
                    and row.status == PipelineProcessState.RUNNING
                    and heartbeat_thread is not None
                    and heartbeat_thread.is_alive()
                ):
                    break
                time.sleep(0.05)

            assert row is not None
            assert row.status == PipelineProcessState.RUNNING
            assert row.run_mode == PipelineRunMode.CONTINUOUS
            assert row.host == "run-host"
            assert row.pid == 5555
            assert row.heartbeat_interval_seconds == 1
            assert row.started_at == t0
            assert row.stopped_at is None
            assert heartbeat_thread is not None
            assert heartbeat_thread.is_alive()
            assert heartbeat_thread.daemon is True

            # Advance clock and wait for PipelineHeartbeat thread to tick
            t1 = clock.advance(1)
            hb_deadline = time.monotonic() + 3.0
            refreshed = False
            while time.monotonic() < hb_deadline:
                with session_factory() as session:
                    hb_row = session.get(PipelineProcessStatus, "default")
                if hb_row is not None and hb_row.last_heartbeat_at >= t1:
                    refreshed = True
                    assert hb_row.started_at == t0
                    assert hb_row.status == PipelineProcessState.RUNNING
                    break
                time.sleep(0.05)
            assert refreshed, "PipelineHeartbeat thread did not refresh last_heartbeat_at"

            # Advance clock and trigger graceful stop
            t2 = clock.advance(2)
            service.stop()
            run_thread.join(timeout=5.0)
            heartbeat_thread.join(timeout=5.0)
            assert not run_thread.is_alive()
            assert not heartbeat_thread.is_alive()

            with session_factory() as session:
                stopped_row = session.get(PipelineProcessStatus, "default")
            assert stopped_row is not None
            assert stopped_row.status == PipelineProcessState.STOPPED
            assert stopped_row.run_mode == PipelineRunMode.CONTINUOUS
            assert stopped_row.started_at == t0
            assert stopped_row.stopped_at is not None
            assert stopped_row.stopped_at >= t2
            assert stopped_row.last_heartbeat_at == stopped_row.stopped_at
        finally:
            service.stop()
            run_thread.join(timeout=3.0)

    def test_poller_service_run_once_records_running_then_stopped_on_success(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory,
            clock=clock,
            host="once-host",
            pid=6666,
        )

        observed_during_run: dict[str, object] = {}

        def fake_fetch_new(limit=50):
            with session_factory() as session:
                active_row = session.get(PipelineProcessStatus, "default")
                if active_row is not None:
                    observed_during_run["status"] = active_row.status
                    observed_during_run["run_mode"] = active_row.run_mode
                    observed_during_run["host"] = active_row.host
                    observed_during_run["pid"] = active_row.pid
                    observed_during_run["stopped_at"] = active_row.stopped_at
            clock.advance(5)
            return []

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.side_effect = fake_fetch_new
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            heartbeat_interval_seconds=25,
            telemetry_recorder=recorder,
            clock=clock,
        )

        try:
            stats = service.run_once()
        finally:
            service.stop()

        assert stats.new_stats.inserted == 0
        assert observed_during_run == {
            "status": PipelineProcessState.RUNNING,
            "run_mode": PipelineRunMode.ONCE,
            "host": "once-host",
            "pid": 6666,
            "stopped_at": None,
        }

        with session_factory() as session:
            row = session.get(PipelineProcessStatus, "default")
        assert row is not None
        assert row.status == PipelineProcessState.STOPPED
        assert row.run_mode == PipelineRunMode.ONCE
        assert row.heartbeat_interval_seconds == 25
        assert row.started_at == t0
        assert row.stopped_at == t0 + timedelta(seconds=5)
        assert row.last_heartbeat_at == row.stopped_at

    def test_poller_service_run_once_marks_stopped_and_reraises_on_exception(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory,
            clock=clock,
            host="fail-host",
            pid=7777,
        )

        def failing_fetch_new(limit=50):
            clock.advance(3)
            raise RuntimeError("reddit 503 unavailable")

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.side_effect = failing_fetch_new

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            telemetry_recorder=recorder,
            clock=clock,
        )

        try:
            with pytest.raises(RuntimeError, match="reddit 503 unavailable"):
                service.run_once()
        finally:
            service.stop()

        with session_factory() as session:
            row = session.get(PipelineProcessStatus, "default")
        assert row is not None
        assert row.status == PipelineProcessState.STOPPED
        assert row.run_mode == PipelineRunMode.ONCE
        assert row.host == "fail-host"
        assert row.pid == 7777
        assert row.started_at == t0
        assert row.stopped_at == t0 + timedelta(seconds=3)
        assert row.last_heartbeat_at == row.stopped_at

    def test_poller_service_continues_when_telemetry_db_fails(
        self, session_factory, caplog
    ):
        failing_factory = MagicMock(side_effect=RuntimeError("telemetry DB down"))
        broken_recorder = PipelineTelemetryRecorder(
            failing_factory,
            clock=lambda: datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            host="resilient-host",
            pid=8888,
        )

        sub = RawSubmission(
            id="tel_fail_sub_1",
            title="Resilient Artist - Resilient LP",
            url="https://reddit.com/r/vinylreleases/tel_fail_1",
            created_utc=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            fullname="t3_tel_fail_sub_1",
        )
        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = [sub]
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            heartbeat_interval_seconds=1,
            telemetry_recorder=broken_recorder,
        )

        with caplog.at_level(logging.WARNING, logger="reddit_scrape.telemetry"):
            stats = service.run_once()

        assert stats.new_stats.inserted == 1
        with session_factory() as session:
            saved = session.get(Submission, "t3_tel_fail_sub_1")
            assert saved is not None
        assert any("record_process_start" in r.message for r in caplog.records)
        assert any("record_process_stop" in r.message for r in caplog.records)

        # Also verify continuous run() completes startup and shutdown cleanly on telemetry failure
        mock_adapter.fetch_new.return_value = []
        timer = threading.Timer(0.2, service.stop)
        timer.start()
        try:
            service.run()
        finally:
            timer.join()
            service.stop()

