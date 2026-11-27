"""PostgreSQL integration and unit tests for poll loop telemetry and run history."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from conftest import (
    FakeClock,
    get_loop_state,
    make_raw_submission,
    truncate_all_tables,
    wait_until,
)
from sqlalchemy import select

from reddit_scrape.extraction import ExtractionResult
from reddit_scrape.models import (
    PipelineLoopName,
    PipelineLoopState,
    PipelineLoopStatus,
    PipelineProcessState,
    PipelineProcessStatus,
    PipelineRun,
    PipelineRunMode,
    PipelineRunStatus,
    Submission,
)
from reddit_scrape.service import PollerService
from reddit_scrape.telemetry import PipelineTelemetryRecorder


@pytest.fixture(autouse=True)
def clean_db(request):
    if request.cls and request.cls.__name__ == "TestPipelineTelemetryPollsUnit":
        yield
        return
    engine = request.getfixturevalue("engine")
    truncate_all_tables(engine)
    yield
    truncate_all_tables(engine)


class TestPipelineTelemetryRecorderLoops:
    """PostgreSQL integration tests for PipelineTelemetryRecorder loop & run methods."""

    def test_seed_loop_states_continuous_enabled_vs_disabled_and_once(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        intervals = {
            PipelineLoopName.DISCOVERY_POLL: 300,
            PipelineLoopName.TRACKING_POLL: 900,
            PipelineLoopName.EXTRACTION_WORKER: 2,
            PipelineLoopName.ENRICHMENT_WORKER: 60,
        }

        recorder.seed_loop_states(
            intervals,
            enrichment_enabled=True,
            is_continuous=True,
            now=t0,
        )

        with session_factory() as session:
            rows = {
                row.loop_name: row
                for row in session.scalars(select(PipelineLoopState)).all()
            }
        assert set(rows.keys()) == {
            PipelineLoopName.DISCOVERY_POLL,
            PipelineLoopName.TRACKING_POLL,
            PipelineLoopName.EXTRACTION_WORKER,
            PipelineLoopName.ENRICHMENT_WORKER,
        }
        assert rows[PipelineLoopName.DISCOVERY_POLL].status == PipelineLoopStatus.SLEEPING
        assert rows[PipelineLoopName.DISCOVERY_POLL].interval_seconds == 300
        assert rows[PipelineLoopName.DISCOVERY_POLL].next_run_at is None

        assert rows[PipelineLoopName.TRACKING_POLL].status == PipelineLoopStatus.SLEEPING
        assert rows[PipelineLoopName.TRACKING_POLL].interval_seconds == 900
        assert rows[PipelineLoopName.TRACKING_POLL].next_run_at is None

        assert rows[PipelineLoopName.EXTRACTION_WORKER].status == PipelineLoopStatus.SLEEPING
        assert rows[PipelineLoopName.EXTRACTION_WORKER].interval_seconds == 2
        assert rows[PipelineLoopName.EXTRACTION_WORKER].next_run_at is None

        assert rows[PipelineLoopName.ENRICHMENT_WORKER].status == PipelineLoopStatus.SLEEPING
        assert rows[PipelineLoopName.ENRICHMENT_WORKER].interval_seconds == 60
        assert rows[PipelineLoopName.ENRICHMENT_WORKER].next_run_at == t0 + timedelta(
            seconds=60
        )

        # Re-seed in once mode with enrichment disabled
        t1 = clock.advance(10)
        recorder.seed_loop_states(
            {**intervals, PipelineLoopName.DISCOVERY_POLL: 120},
            enrichment_enabled=False,
            is_continuous=False,
            now=t1,
        )
        with session_factory() as session:
            rows = {
                row.loop_name: row
                for row in session.scalars(select(PipelineLoopState)).all()
            }
        assert rows[PipelineLoopName.DISCOVERY_POLL].interval_seconds == 120
        assert rows[PipelineLoopName.ENRICHMENT_WORKER].status == PipelineLoopStatus.DISABLED
        assert rows[PipelineLoopName.ENRICHMENT_WORKER].next_run_at is None

        # Re-seed in once mode with enrichment enabled -> next_run_at is None
        recorder.seed_loop_states(
            intervals,
            enrichment_enabled=True,
            is_continuous=False,
            now=t1,
        )
        with session_factory() as session:
            enrichment_row = session.get(
                PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER
            )
        assert enrichment_row is not None
        assert enrichment_row.status == PipelineLoopStatus.SLEEPING
        assert enrichment_row.next_run_at is None

    def test_record_loop_start_completion_error_recovery_and_stop(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory,
            clock=clock,
            host="poll-host",
            pid=9876,
        )
        recorder.record_process_start(
            run_mode=PipelineRunMode.CONTINUOUS,
            heartbeat_interval_seconds=30,
            started_at=t0,
        )
        recorder.seed_loop_states(
            {
                PipelineLoopName.DISCOVERY_POLL: 300,
                PipelineLoopName.TRACKING_POLL: 900,
                PipelineLoopName.EXTRACTION_WORKER: 2,
                PipelineLoopName.ENRICHMENT_WORKER: 60,
            },
            enrichment_enabled=False,
            is_continuous=True,
            now=t0,
        )

        # Start discovery poll -> running with next_run_at=None
        t_start = clock.advance(1)
        recorder.record_loop_start(
            PipelineLoopName.DISCOVERY_POLL, started_at=t_start
        )
        with session_factory() as session:
            state = session.get(PipelineLoopState, PipelineLoopName.DISCOVERY_POLL)
        assert state is not None
        assert state.status == PipelineLoopStatus.RUNNING
        assert state.interval_seconds == 300
        assert state.last_started_at == t_start
        assert state.next_run_at is None

        # Complete with error
        t_err = clock.advance(2.5)
        next_retry = t_err + timedelta(seconds=300)
        recorder.record_loop_completion(
            PipelineLoopName.DISCOVERY_POLL,
            status=PipelineLoopStatus.ERROR,
            started_at=t_start,
            completed_at=t_err,
            stats=None,
            error_message="reddit 503",
            next_run_at=next_retry,
            record_run=True,
        )

        with session_factory() as session:
            state = session.get(PipelineLoopState, PipelineLoopName.DISCOVERY_POLL)
            runs = session.scalars(
                select(PipelineRun).order_by(PipelineRun.id.asc())
            ).all()
        assert state is not None
        assert state.status == PipelineLoopStatus.ERROR
        assert state.last_completed_at == t_err
        assert state.next_run_at == next_retry
        assert state.last_duration_ms == 2500
        assert state.last_run_stats is None
        assert state.last_error == "reddit 503"
        assert len(runs) == 1
        assert runs[0].loop_name == PipelineLoopName.DISCOVERY_POLL
        assert runs[0].status == PipelineRunStatus.ERROR
        assert runs[0].duration_ms == 2500
        assert runs[0].error_message == "reddit 503"
        assert runs[0].host == "poll-host"
        assert runs[0].pid == 9876

        # Subsequent success clears last_error and records success run
        t_succ_start = clock.advance(300)
        recorder.record_loop_start(
            PipelineLoopName.DISCOVERY_POLL, started_at=t_succ_start
        )
        t_succ_end = clock.advance(1.25)
        next_succ = t_succ_end + timedelta(seconds=300)
        recorder.record_loop_completion(
            PipelineLoopName.DISCOVERY_POLL,
            status=PipelineLoopStatus.SLEEPING,
            started_at=t_succ_start,
            completed_at=t_succ_end,
            stats={"inserted": 2, "updated": 0, "skipped": 1},
            next_run_at=next_succ,
            record_run=True,
        )

        with session_factory() as session:
            state = session.get(PipelineLoopState, PipelineLoopName.DISCOVERY_POLL)
            runs = session.scalars(
                select(PipelineRun).order_by(PipelineRun.id.asc())
            ).all()
        assert state is not None
        assert state.status == PipelineLoopStatus.SLEEPING
        assert state.last_error is None
        assert state.last_run_stats == {"inserted": 2, "updated": 0, "skipped": 1}
        assert state.last_duration_ms == 1250
        assert len(runs) == 2
        assert runs[1].status == PipelineRunStatus.SUCCESS
        assert runs[1].stats == {"inserted": 2, "updated": 0, "skipped": 1}
        assert runs[1].error_message is None

        # Graceful stop transitions non-disabled rows to stopped, clears next_run_at,
        # preserves last_* metrics, and leaves disabled enrichment_worker untouched
        t_stop = clock.advance(5)
        recorder.record_process_stop(stopped_at=t_stop)

        with session_factory() as session:
            proc = session.get(PipelineProcessStatus, "default")
            states = {
                row.loop_name: row
                for row in session.scalars(select(PipelineLoopState)).all()
            }
        assert proc is not None
        assert proc.status == PipelineProcessState.STOPPED

        disc = states[PipelineLoopName.DISCOVERY_POLL]
        assert disc.status == PipelineLoopStatus.STOPPED
        assert disc.next_run_at is None
        assert disc.last_started_at == t_succ_start
        assert disc.last_completed_at == t_succ_end
        assert disc.last_duration_ms == 1250
        assert disc.last_run_stats == {"inserted": 2, "updated": 0, "skipped": 1}
        assert disc.last_error is None

        assert states[PipelineLoopName.TRACKING_POLL].status == PipelineLoopStatus.STOPPED
        assert states[PipelineLoopName.EXTRACTION_WORKER].status == PipelineLoopStatus.STOPPED
        assert states[PipelineLoopName.ENRICHMENT_WORKER].status == PipelineLoopStatus.DISABLED
        assert states[PipelineLoopName.ENRICHMENT_WORKER].next_run_at is None


class TestPipelineTelemetryPollsUnit:
    """Unit tests for loop telemetry failure isolation."""

    def test_loop_telemetry_methods_catch_db_errors_and_log_warning(self, caplog):
        failing_factory = MagicMock(side_effect=RuntimeError("db down"))
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        recorder = PipelineTelemetryRecorder(failing_factory, clock=lambda: t0)

        with caplog.at_level(logging.WARNING, logger="reddit_scrape.telemetry"):
            recorder.seed_loop_states(
                {
                    PipelineLoopName.DISCOVERY_POLL: 300,
                    PipelineLoopName.TRACKING_POLL: 900,
                    PipelineLoopName.EXTRACTION_WORKER: 2,
                    PipelineLoopName.ENRICHMENT_WORKER: 60,
                },
                enrichment_enabled=True,
                is_continuous=True,
                now=t0,
            )
            recorder.record_loop_start(PipelineLoopName.DISCOVERY_POLL, started_at=t0)
            recorder.record_loop_completion(
                PipelineLoopName.DISCOVERY_POLL,
                status=PipelineLoopStatus.SLEEPING,
                started_at=t0,
                completed_at=t0 + timedelta(seconds=1),
                stats={"inserted": 0, "updated": 0, "skipped": 0},
                next_run_at=t0 + timedelta(seconds=301),
            )

        assert len(caplog.records) == 3
        assert all(r.levelno == logging.WARNING for r in caplog.records)


class TestPollerServicePollTelemetryIntegration:
    """PostgreSQL integration tests for PollerService discovery/tracking poll telemetry."""

    def test_run_seeds_all_loops_tracks_polls_and_aligns_next_run_at(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory, clock=clock, host="svc-host", pid=4321
        )

        observed_during_new: dict[str, object] = {}
        new_calls = 0

        def fake_fetch_new(limit=50):
            nonlocal new_calls
            new_calls += 1
            with session_factory() as session:
                row = session.get(PipelineLoopState, PipelineLoopName.DISCOVERY_POLL)
                if row is not None:
                    observed_during_new[f"call_{new_calls}"] = (
                        row.status,
                        row.next_run_at,
                        row.last_started_at,
                    )
            clock.advance(2)
            if new_calls == 1:
                return [
                    make_raw_submission(
                        "disc_1",
                        title="Artist - Album [LP]",
                        created_utc=t0,
                    )
                ]
            return []

        def fake_fetch_hot(limit=50):
            clock.advance(1)
            return []

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.side_effect = fake_fetch_new
        mock_adapter.fetch_hot.side_effect = fake_fetch_hot

        mock_worker = MagicMock()
        mock_worker.extract_batch.side_effect = lambda batch, batch_size=5: {
            sub_id: ExtractionResult(releases=[]) for sub_id, _, _ in batch
        }

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=mock_worker,
            poll_interval_seconds=60,
            hot_poll_interval_seconds=300,
            enrichment_interval_seconds=45,
            heartbeat_interval_seconds=30,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()

        try:
            assert wait_until(
                lambda: (
                    (s := get_loop_state(session_factory, PipelineLoopName.TRACKING_POLL))
                    is not None
                    and s.status == PipelineLoopStatus.SLEEPING
                    and s.last_completed_at is not None
                    and (e := get_loop_state(session_factory, PipelineLoopName.EXTRACTION_WORKER))
                    is not None
                    and e.status == PipelineLoopStatus.SLEEPING
                )
            ), "Initial polls did not complete"

            with session_factory() as session:
                states = {
                    row.loop_name: row
                    for row in session.scalars(select(PipelineLoopState)).all()
                }
                runs = session.scalars(
                    select(PipelineRun)
                    .where(
                        PipelineRun.loop_name.in_(
                            [
                                PipelineLoopName.DISCOVERY_POLL,
                                PipelineLoopName.TRACKING_POLL,
                            ]
                        )
                    )
                    .order_by(PipelineRun.id.asc())
                ).all()

            assert observed_during_new["call_1"] == (
                PipelineLoopStatus.RUNNING,
                None,
                t0,
            )

            disc = states[PipelineLoopName.DISCOVERY_POLL]
            assert disc.status == PipelineLoopStatus.SLEEPING
            assert disc.interval_seconds == 60
            assert disc.last_started_at == t0
            assert disc.last_completed_at == t0 + timedelta(seconds=2)
            assert disc.last_duration_ms == 2000
            assert disc.last_run_stats == {"inserted": 1, "updated": 0, "skipped": 0}
            assert disc.last_error is None
            assert disc.next_run_at == disc.last_completed_at + timedelta(seconds=60)

            track = states[PipelineLoopName.TRACKING_POLL]
            assert track.status == PipelineLoopStatus.SLEEPING
            assert track.interval_seconds == 300
            assert track.last_started_at == t0 + timedelta(seconds=2)
            assert track.last_completed_at == t0 + timedelta(seconds=3)
            assert track.last_duration_ms == 1000
            assert track.last_run_stats == {"inserted": 0, "updated": 0, "skipped": 0}
            assert track.next_run_at == track.last_completed_at + timedelta(seconds=300)

            ext = states[PipelineLoopName.EXTRACTION_WORKER]
            assert ext.status == PipelineLoopStatus.SLEEPING
            assert ext.interval_seconds == 2
            assert ext.next_run_at is None

            enr = states[PipelineLoopName.ENRICHMENT_WORKER]
            assert enr.status == PipelineLoopStatus.DISABLED
            assert enr.interval_seconds == 45
            assert enr.next_run_at is None

            # Both initial polls logged PipelineRun rows (including 0-item tracking poll)
            assert len(runs) == 2
            assert runs[0].loop_name == PipelineLoopName.DISCOVERY_POLL
            assert runs[0].status == PipelineRunStatus.SUCCESS
            assert runs[0].stats == {"inserted": 1, "updated": 0, "skipped": 0}
            assert runs[1].loop_name == PipelineLoopName.TRACKING_POLL
            assert runs[1].status == PipelineRunStatus.SUCCESS
            assert runs[1].stats == {"inserted": 0, "updated": 0, "skipped": 0}

            # Advance clock just before disc.next_run_at -> should NOT poll yet
            clock.current = disc.next_run_at - timedelta(seconds=1)
            time.sleep(1.2)
            assert new_calls == 1

            # Advance clock to disc.next_run_at -> scheduled discovery_poll triggers!
            expected_sched_start = disc.next_run_at
            clock.current = expected_sched_start
            assert wait_until(
                lambda: new_calls >= 2
                and (
                    (r := get_loop_state(session_factory, PipelineLoopName.DISCOVERY_POLL))
                    is not None
                    and r.status == PipelineLoopStatus.SLEEPING
                    and r.last_started_at == expected_sched_start
                )
            ), "Scheduled discovery_poll did not trigger on aligned next_run_at"

            with session_factory() as session:
                disc2 = session.get(PipelineLoopState, PipelineLoopName.DISCOVERY_POLL)
                runs2 = session.scalars(
                    select(PipelineRun)
                    .where(
                        PipelineRun.loop_name.in_(
                            [
                                PipelineLoopName.DISCOVERY_POLL,
                                PipelineLoopName.TRACKING_POLL,
                            ]
                        )
                    )
                    .order_by(PipelineRun.id.asc())
                ).all()
            assert disc2 is not None
            assert disc2.last_started_at == expected_sched_start
            assert disc2.last_completed_at == expected_sched_start + timedelta(seconds=2)
            assert disc2.next_run_at == disc2.last_completed_at + timedelta(seconds=60)
            assert disc2.last_run_stats == {"inserted": 0, "updated": 0, "skipped": 0}
            assert len(runs2) == 3
            assert runs2[2].loop_name == PipelineLoopName.DISCOVERY_POLL
            assert runs2[2].status == PipelineRunStatus.SUCCESS

            # Graceful shutdown transitions non-disabled loops to stopped, clears next_run_at,
            # preserves last_* fields, and leaves disabled enrichment_worker untouched
            service.stop()
            run_thread.join(timeout=5.0)
            assert not run_thread.is_alive()

            with session_factory() as session:
                stopped_states = {
                    row.loop_name: row
                    for row in session.scalars(select(PipelineLoopState)).all()
                }
            disc_stopped = stopped_states[PipelineLoopName.DISCOVERY_POLL]
            assert disc_stopped.status == PipelineLoopStatus.STOPPED
            assert disc_stopped.next_run_at is None
            assert disc_stopped.last_started_at == expected_sched_start
            assert disc_stopped.last_completed_at == disc2.last_completed_at
            assert disc_stopped.last_duration_ms == 2000
            assert disc_stopped.last_run_stats == {"inserted": 0, "updated": 0, "skipped": 0}

            assert stopped_states[PipelineLoopName.TRACKING_POLL].status == PipelineLoopStatus.STOPPED
            assert stopped_states[PipelineLoopName.TRACKING_POLL].next_run_at is None
            assert stopped_states[PipelineLoopName.EXTRACTION_WORKER].status == PipelineLoopStatus.STOPPED
            assert stopped_states[PipelineLoopName.ENRICHMENT_WORKER].status == PipelineLoopStatus.DISABLED
        finally:
            service.stop()
            run_thread.join(timeout=3.0)

    def test_run_seeds_enabled_enrichment_worker_with_next_run_at(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []
        mock_enrichment = MagicMock()
        mock_enrichment.process_pending_enrichment.return_value = 0

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=mock_enrichment,
            enrichment_interval_seconds=90,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()
        try:
            assert wait_until(
                lambda: get_loop_state(
                    session_factory, PipelineLoopName.ENRICHMENT_WORKER
                )
                is not None
            )
            with session_factory() as session:
                enr = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
            assert enr is not None
            assert enr.status == PipelineLoopStatus.SLEEPING
            assert enr.interval_seconds == 90
            assert enr.next_run_at == t0 + timedelta(seconds=90)

            service.stop()
            run_thread.join(timeout=5.0)

            with session_factory() as session:
                enr_stopped = session.get(
                    PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER
                )
            assert enr_stopped is not None
            assert enr_stopped.status == PipelineLoopStatus.STOPPED
            assert enr_stopped.next_run_at is None
        finally:
            service.stop()
            run_thread.join(timeout=3.0)

    def test_run_poll_error_records_error_recovers_and_preserves_last_error_on_stop(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        new_attempt = 0

        def fake_fetch_new(limit=50):
            nonlocal new_attempt
            new_attempt += 1
            clock.advance(2)
            if new_attempt == 1:
                raise RuntimeError("reddit 503 temporary")
            return [
                make_raw_submission(
                    "rec_1",
                    title="Recovered Artist - Album [LP]",
                    created_utc=t0,
                )
            ]

        def fake_fetch_hot(limit=50):
            clock.advance(1)
            raise RuntimeError("hot feed failed")

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.side_effect = fake_fetch_new
        mock_adapter.fetch_hot.side_effect = fake_fetch_hot

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            poll_interval_seconds=30,
            hot_poll_interval_seconds=600,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()
        try:
            # Wait for both initial polls (both error) to complete
            assert wait_until(
                lambda: (
                    (r := get_loop_state(session_factory, PipelineLoopName.TRACKING_POLL))
                    is not None
                    and r.status == PipelineLoopStatus.ERROR
                )
            )
            with session_factory() as session:
                disc = session.get(PipelineLoopState, PipelineLoopName.DISCOVERY_POLL)
                track = session.get(PipelineLoopState, PipelineLoopName.TRACKING_POLL)
                runs = session.scalars(
                    select(PipelineRun).order_by(PipelineRun.id.asc())
                ).all()

            assert disc is not None
            assert disc.status == PipelineLoopStatus.ERROR
            assert disc.last_error == "reddit 503 temporary"
            assert disc.next_run_at == disc.last_completed_at + timedelta(seconds=30)
            assert track is not None
            assert track.status == PipelineLoopStatus.ERROR
            assert track.last_error == "hot feed failed"
            assert len(runs) == 2
            assert runs[0].status == PipelineRunStatus.ERROR
            assert runs[0].error_message == "reddit 503 temporary"
            assert runs[1].status == PipelineRunStatus.ERROR
            assert runs[1].error_message == "hot feed failed"

            # Advance clock to trigger scheduled discovery_poll recovery (tracking_poll stays in error)
            clock.current = disc.next_run_at
            assert wait_until(
                lambda: (
                    (r := get_loop_state(session_factory, PipelineLoopName.DISCOVERY_POLL))
                    is not None
                    and r.status == PipelineLoopStatus.SLEEPING
                )
            )
            with session_factory() as session:
                disc_rec = session.get(PipelineLoopState, PipelineLoopName.DISCOVERY_POLL)
                runs_after = session.scalars(
                    select(PipelineRun)
                    .where(
                        PipelineRun.loop_name.in_(
                            [
                                PipelineLoopName.DISCOVERY_POLL,
                                PipelineLoopName.TRACKING_POLL,
                            ]
                        )
                    )
                    .order_by(PipelineRun.id.asc())
                ).all()
            assert disc_rec is not None
            assert disc_rec.status == PipelineLoopStatus.SLEEPING
            assert disc_rec.last_error is None
            assert disc_rec.last_run_stats == {"inserted": 1, "updated": 0, "skipped": 0}
            assert len(runs_after) == 3
            assert runs_after[2].status == PipelineRunStatus.SUCCESS

            # Shutdown while tracking_poll is in error -> transitions to stopped, preserves last_error
            service.stop()
            run_thread.join(timeout=5.0)
            with session_factory() as session:
                track_stopped = session.get(
                    PipelineLoopState, PipelineLoopName.TRACKING_POLL
                )
            assert track_stopped is not None
            assert track_stopped.status == PipelineLoopStatus.STOPPED
            assert track_stopped.next_run_at is None
            assert track_stopped.last_error == "hot feed failed"
        finally:
            service.stop()
            run_thread.join(timeout=3.0)

    def test_run_once_records_loop_telemetry_and_handles_fail_fast_error(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory, clock=clock, host="once-host", pid=1010
        )

        observed_next_run: list[datetime | None] = []

        def fake_fetch_new(limit=50):
            clock.advance(2)
            return [
                make_raw_submission(
                    "once_1",
                    title="Once Artist - Album [LP]",
                    created_utc=t0,
                )
            ]

        def fake_fetch_hot(limit=50):
            with session_factory() as session:
                disc = session.get(PipelineLoopState, PipelineLoopName.DISCOVERY_POLL)
                if disc is not None:
                    observed_next_run.append(disc.next_run_at)
            clock.advance(1)
            return []

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.side_effect = fake_fetch_new
        mock_adapter.fetch_hot.side_effect = fake_fetch_hot

        mock_worker = MagicMock()
        mock_worker.extract_batch.side_effect = lambda batch, batch_size=5: {
            sub_id: ExtractionResult(releases=[]) for sub_id, _, _ in batch
        }

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=mock_worker,
            poll_interval_seconds=300,
            hot_poll_interval_seconds=900,
            telemetry_recorder=recorder,
            clock=clock,
        )
        try:
            stats = service.run_once()
        finally:
            service.stop()

        assert stats.new_stats.inserted == 1
        assert observed_next_run == [None]

        with session_factory() as session:
            proc = session.get(PipelineProcessStatus, "default")
            states = {
                row.loop_name: row
                for row in session.scalars(select(PipelineLoopState)).all()
            }
            runs = session.scalars(
                select(PipelineRun)
                .where(
                    PipelineRun.loop_name.in_(
                        [
                            PipelineLoopName.DISCOVERY_POLL,
                            PipelineLoopName.TRACKING_POLL,
                        ]
                    )
                )
                .order_by(PipelineRun.id.asc())
            ).all()

        assert proc is not None
        assert proc.status == PipelineProcessState.STOPPED
        assert proc.run_mode == PipelineRunMode.ONCE
        assert states[PipelineLoopName.DISCOVERY_POLL].status == PipelineLoopStatus.STOPPED
        assert states[PipelineLoopName.DISCOVERY_POLL].next_run_at is None
        assert states[PipelineLoopName.DISCOVERY_POLL].last_run_stats == {
            "inserted": 1,
            "updated": 0,
            "skipped": 0,
        }
        assert states[PipelineLoopName.TRACKING_POLL].status == PipelineLoopStatus.STOPPED
        assert states[PipelineLoopName.TRACKING_POLL].next_run_at is None
        assert states[PipelineLoopName.TRACKING_POLL].last_run_stats == {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
        }
        assert states[PipelineLoopName.ENRICHMENT_WORKER].status == PipelineLoopStatus.DISABLED
        assert len(runs) == 2
        assert runs[0].loop_name == PipelineLoopName.DISCOVERY_POLL
        assert runs[0].status == PipelineRunStatus.SUCCESS
        assert runs[1].loop_name == PipelineLoopName.TRACKING_POLL
        assert runs[1].status == PipelineRunStatus.SUCCESS

        # Now verify fail-fast error in run_once() when tracking_poll raises
        def failing_hot(limit=50):
            clock.advance(4)
            raise RuntimeError("tracking boom")

        mock_adapter.fetch_hot.side_effect = failing_hot
        mock_adapter.fetch_new.side_effect = lambda limit=50: (clock.advance(1), [])[1]

        service_fail = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            telemetry_recorder=recorder,
            clock=clock,
        )
        try:
            with pytest.raises(RuntimeError, match="tracking boom"):
                service_fail.run_once()
        finally:
            service_fail.stop()

        with session_factory() as session:
            proc_fail = session.get(PipelineProcessStatus, "default")
            track_fail = session.get(PipelineLoopState, PipelineLoopName.TRACKING_POLL)
            all_runs = session.scalars(
                select(PipelineRun).order_by(PipelineRun.id.asc())
            ).all()
        assert proc_fail is not None
        assert proc_fail.status == PipelineProcessState.STOPPED
        assert track_fail is not None
        assert track_fail.status == PipelineLoopStatus.STOPPED
        assert track_fail.next_run_at is None
        assert track_fail.last_error == "tracking boom"
        assert all_runs[-1].loop_name == PipelineLoopName.TRACKING_POLL
        assert all_runs[-1].status == PipelineRunStatus.ERROR
        assert all_runs[-1].error_message == "tracking boom"

    def test_poll_telemetry_db_failure_does_not_abort_polling(
        self, session_factory, caplog
    ):
        failing_factory = MagicMock(side_effect=RuntimeError("telemetry DB offline"))
        broken_recorder = PipelineTelemetryRecorder(
            failing_factory,
            clock=lambda: datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        sub = make_raw_submission(
            "resilient_poll_1",
            title="Resilient Band - Resilient Album",
            created_utc=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = [sub]
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            telemetry_recorder=broken_recorder,
        )

        with caplog.at_level(logging.WARNING, logger="reddit_scrape.telemetry"):
            stats = service.run_once()

        assert stats.new_stats.inserted == 1
        with session_factory() as session:
            assert session.get(Submission, "t3_resilient_poll_1") is not None
        assert any("seed_loop_states" in r.message for r in caplog.records)
        assert any("record_loop_start" in r.message for r in caplog.records)
        assert any("record_loop_completion" in r.message for r in caplog.records)
