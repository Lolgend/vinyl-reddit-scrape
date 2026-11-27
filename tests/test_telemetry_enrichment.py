"""PostgreSQL integration and unit tests for Enrichment Queue telemetry."""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from conftest import FakeClock, get_loop_state, truncate_all_tables, wait_until
from sqlalchemy import select

from reddit_scrape.enrichment import CatalogEnrichmentService
from reddit_scrape.models import (
    Album,
    PipelineLoopName,
    PipelineLoopState,
    PipelineLoopStatus,
    PipelineProcessState,
    PipelineProcessStatus,
    PipelineRun,
    PipelineRunMode,
    PipelineRunStatus,
)
from reddit_scrape.service import PollerService
from reddit_scrape.telemetry import PipelineTelemetryRecorder


@pytest.fixture(autouse=True)
def clean_db(request):
    if request.cls and request.cls.__name__ == "TestEnrichmentTelemetryUnit":
        yield
        return
    engine = request.getfixturevalue("engine")
    truncate_all_tables(engine)
    yield
    truncate_all_tables(engine)


class TestTelemetryRecorderRefreshLoopNextRun:
    """PostgreSQL integration tests for PipelineTelemetryRecorder.refresh_loop_next_run."""

    def test_refresh_loop_next_run_updates_status_next_run_and_clears_error_without_run_row(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

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

        t_run_start = clock.advance(10)
        t_run_end = clock.advance(2)
        recorder.record_loop_completion(
            PipelineLoopName.ENRICHMENT_WORKER,
            status=PipelineLoopStatus.ERROR,
            started_at=t_run_start,
            completed_at=t_run_end,
            stats={"enriched": 3},
            error_message="transient db error",
            next_run_at=t_run_end + timedelta(seconds=60),
            record_run=False,
        )

        next_wakeup = clock.advance(60) + timedelta(seconds=60)
        recorder.refresh_loop_next_run(
            PipelineLoopName.ENRICHMENT_WORKER,
            next_run_at=next_wakeup,
            clear_error=True,
        )

        with session_factory() as session:
            state = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
            runs = session.scalars(select(PipelineRun)).all()

        assert state is not None
        assert state.status == PipelineLoopStatus.SLEEPING
        assert state.interval_seconds == 60
        assert state.next_run_at == next_wakeup
        assert state.last_error is None
        assert state.last_started_at == t_run_start
        assert state.last_completed_at == t_run_end
        assert state.last_duration_ms == 2000
        assert state.last_run_stats == {"enriched": 3}
        assert runs == []

        # Verify clear_error=False preserves existing last_error
        recorder.record_loop_completion(
            PipelineLoopName.ENRICHMENT_WORKER,
            status=PipelineLoopStatus.ERROR,
            started_at=t_run_start,
            completed_at=t_run_end,
            stats={"enriched": 3},
            error_message="kept error",
            next_run_at=next_wakeup,
            record_run=False,
        )
        next_wakeup_2 = next_wakeup + timedelta(seconds=60)
        recorder.refresh_loop_next_run(
            PipelineLoopName.ENRICHMENT_WORKER,
            next_run_at=next_wakeup_2,
            clear_error=False,
        )
        with session_factory() as session:
            state_kept = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
        assert state_kept is not None
        assert state_kept.status == PipelineLoopStatus.SLEEPING
        assert state_kept.next_run_at == next_wakeup_2
        assert state_kept.last_error == "kept error"

        # Verify completed_at & duration_ms update without clobbering last_run_stats or adding runs
        t_completed = clock.advance(5)
        next_wakeup_3 = t_completed + timedelta(seconds=60)
        recorder.refresh_loop_next_run(
            PipelineLoopName.ENRICHMENT_WORKER,
            next_run_at=next_wakeup_3,
            clear_error=True,
            completed_at=t_completed,
            duration_ms=420,
        )
        with session_factory() as session:
            state_comp = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
            runs_after = session.scalars(select(PipelineRun)).all()
        assert state_comp is not None
        assert state_comp.last_completed_at == t_completed
        assert state_comp.last_duration_ms == 420
        assert state_comp.last_run_stats == {"enriched": 3}
        assert state_comp.last_error is None
        assert runs_after == []


class TestEnrichmentWorkerTelemetryIntegration:
    """PostgreSQL integration tests for Enrichment Queue telemetry in PollerService."""

    def test_run_disabled_enrichment_does_not_start_thread_and_preserves_disabled(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=None,
            enrichment_interval_seconds=1,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()
        try:
            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.TRACKING_POLL
                        )
                    )
                    is not None
                    and r.last_completed_at is not None
                )
            )

            thread_names = {t.name for t in threading.enumerate()}
            assert "DiscogsEnrichmentWorker" not in thread_names

            with session_factory() as session:
                enr = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
                enr_runs = session.scalars(
                    select(PipelineRun).where(
                        PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER
                    )
                ).all()

            assert enr is not None
            assert enr.status == PipelineLoopStatus.DISABLED
            assert enr.next_run_at is None
            assert enr_runs == []

            service.stop()
            run_thread.join(timeout=5.0)
            assert not run_thread.is_alive()

            with session_factory() as session:
                enr_after = session.get(
                    PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER
                )
                enr_runs_after = session.scalars(
                    select(PipelineRun).where(
                        PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER
                    )
                ).all()

            assert enr_after is not None
            assert enr_after.status == PipelineLoopStatus.DISABLED
            assert enr_after.next_run_at is None
            assert enr_runs_after == []
        finally:
            service.stop()
            run_thread.join(timeout=3.0)

    def test_run_idle_enrichment_wakeup_refreshes_next_run_without_running_flap_or_run_row(
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
            enrichment_interval_seconds=60,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()
        try:
            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.ENRICHMENT_WORKER
                        )
                    )
                    is not None
                    and r.next_run_at == t0 + timedelta(seconds=60)
                )
            )
            # Wait for initial polls to finish so clock doesn't race
            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.TRACKING_POLL
                        )
                    )
                    is not None
                    and r.last_completed_at is not None
                )
            )

            # Simulate a prior loop-level error to verify idle refresh clears it
            with session_factory() as session, session.begin():
                row = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
                assert row is not None
                row.last_error = "stale loop error"

            # Advance clock and trigger an idle enrichment wakeup (no Album rows in DB)
            t_wakeup = clock.advance(60)
            service.trigger_enrichment()

            expected_next_run = t_wakeup + timedelta(seconds=60)
            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.ENRICHMENT_WORKER
                        )
                    )
                    is not None
                    and r.next_run_at == expected_next_run
                )
            )

            with session_factory() as session:
                enr = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
                enr_runs = session.scalars(
                    select(PipelineRun).where(
                        PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER
                    )
                ).all()

            assert enr is not None
            assert enr.status == PipelineLoopStatus.SLEEPING
            assert enr.next_run_at == expected_next_run
            assert enr.last_error is None
            assert enr.last_started_at is None
            assert enr.last_completed_at is None
            assert enr.last_duration_ms is None
            assert enr.last_run_stats is None
            assert enr_runs == []
            mock_enrichment.process_pending_enrichment.assert_not_called()
        finally:
            service.stop()
            run_thread.join(timeout=3.0)

    def test_run_active_enrichment_logs_run_and_preserves_metrics_on_shutdown(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory, clock=clock, host="enr-host", pid=4567
        )

        album_id = uuid.uuid4()
        with session_factory() as session, session.begin():
            album = Album(
                id=album_id,
                combined_artist="Boards of Canada",
                title="Music Has the Right to Children",
                fingerprint=f"fp-{album_id}",
                discogs_status="pending",
            )
            session.add(album)

        observed_running: list[tuple[str, datetime | None, datetime | None]] = []

        def fake_process_pending(limit=None, shutdown_event=None):
            with session_factory() as session:
                state = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
                if state is not None:
                    observed_running.append(
                        (state.status, state.last_started_at, state.next_run_at)
                    )
            clock.advance(3)
            with session_factory() as session, session.begin():
                row = session.get(Album, album_id)
                assert row is not None
                row.discogs_status = "matched"
            return 2

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []
        mock_enrichment = MagicMock()
        mock_enrichment.process_pending_enrichment.side_effect = fake_process_pending

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=mock_enrichment,
            enrichment_interval_seconds=60,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()
        try:
            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.TRACKING_POLL
                        )
                    )
                    is not None
                    and r.last_completed_at is not None
                )
            )

            t_wakeup = clock.advance(10)
            service.trigger_enrichment()

            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.ENRICHMENT_WORKER
                        )
                    )
                    is not None
                    and r.last_completed_at is not None
                )
            )

            assert observed_running == [
                (PipelineLoopStatus.RUNNING, t_wakeup, None)
            ]

            with session_factory() as session:
                enr = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
                enr_runs = session.scalars(
                    select(PipelineRun)
                    .where(PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER)
                    .order_by(PipelineRun.id.asc())
                ).all()

            expected_completed = t_wakeup + timedelta(seconds=3)
            expected_next = expected_completed + timedelta(seconds=60)
            assert enr is not None
            assert enr.status == PipelineLoopStatus.SLEEPING
            assert enr.last_started_at == t_wakeup
            assert enr.last_completed_at == expected_completed
            assert enr.last_duration_ms == 3000
            assert enr.last_run_stats == {"enriched": 2}
            assert enr.last_error is None
            assert enr.next_run_at == expected_next

            assert len(enr_runs) == 1
            assert enr_runs[0].status == PipelineRunStatus.SUCCESS
            assert enr_runs[0].started_at == t_wakeup
            assert enr_runs[0].completed_at == expected_completed
            assert enr_runs[0].duration_ms == 3000
            assert enr_runs[0].stats == {"enriched": 2}
            assert enr_runs[0].error_message is None
            assert enr_runs[0].host == "enr-host"
            assert enr_runs[0].pid == 4567

            service.stop()
            run_thread.join(timeout=5.0)

            with session_factory() as session:
                enr_stopped = session.get(
                    PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER
                )
            assert enr_stopped is not None
            assert enr_stopped.status == PipelineLoopStatus.STOPPED
            assert enr_stopped.next_run_at is None
            assert enr_stopped.last_started_at == t_wakeup
            assert enr_stopped.last_completed_at == expected_completed
            assert enr_stopped.last_duration_ms == 3000
            assert enr_stopped.last_run_stats == {"enriched": 2}
        finally:
            service.stop()
            run_thread.join(timeout=3.0)

    def test_run_per_album_failure_stays_on_album_without_loop_error_or_run_row(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        album_id = uuid.uuid4()
        with session_factory() as session, session.begin():
            album = Album(
                id=album_id,
                combined_artist="Failing Artist",
                title="Failing Album",
                fingerprint=f"fp-{album_id}",
                discogs_status="pending",
            )
            session.add(album)

        mock_discogs = MagicMock()
        mock_discogs.search.side_effect = RuntimeError("discogs 502 bad gateway")
        real_enrichment = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=mock_discogs,
        )

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=real_enrichment,
            enrichment_interval_seconds=60,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()
        try:
            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.TRACKING_POLL
                        )
                    )
                    is not None
                    and r.last_completed_at is not None
                )
            )

            t_wakeup = clock.advance(10)
            service.trigger_enrichment()

            expected_next = t_wakeup + timedelta(seconds=60)
            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.ENRICHMENT_WORKER
                        )
                    )
                    is not None
                    and r.next_run_at == expected_next
                )
            )

            with session_factory() as session:
                album_after = session.get(Album, album_id)
                enr = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
                enr_runs = session.scalars(
                    select(PipelineRun).where(
                        PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER
                    )
                ).all()

            assert album_after is not None
            assert album_after.discogs_status == "failed"
            assert album_after.enrichment_error == "discogs 502 bad gateway"
            assert enr is not None
            assert enr.status == PipelineLoopStatus.SLEEPING
            assert enr.last_error is None
            assert enr.next_run_at == expected_next
            assert enr_runs == []
        finally:
            service.stop()
            run_thread.join(timeout=3.0)

    def test_run_active_enrichment_zero_enriched_sets_completed_at_and_preserves_stats(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        album_id = uuid.uuid4()
        with session_factory() as session, session.begin():
            session.add(
                Album(
                    id=album_id,
                    combined_artist="Zero Enriched Artist",
                    title="Zero Enriched Album",
                    fingerprint=f"fp-{album_id}",
                    discogs_status="pending",
                )
            )

        def failing_search(*args, **kwargs):
            clock.advance(2)
            raise RuntimeError("discogs 502 bad gateway")

        mock_discogs = MagicMock()
        mock_discogs.search.side_effect = failing_search
        real_enrichment = CatalogEnrichmentService(
            session_factory=session_factory,
            discogs_client=mock_discogs,
        )

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=real_enrichment,
            enrichment_interval_seconds=60,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()
        try:
            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.TRACKING_POLL
                        )
                    )
                    is not None
                    and r.last_completed_at is not None
                )
            )

            # Seed prior completion metrics so we verify last_run_stats is preserved
            t_prior_start = t0 - timedelta(minutes=2)
            t_prior_end = t_prior_start + timedelta(seconds=1)
            recorder.record_loop_completion(
                PipelineLoopName.ENRICHMENT_WORKER,
                status=PipelineLoopStatus.SLEEPING,
                started_at=t_prior_start,
                completed_at=t_prior_end,
                stats={"enriched": 4},
                next_run_at=t0 + timedelta(seconds=60),
                record_run=False,
            )

            t_wakeup = clock.advance(10)
            service.trigger_enrichment()

            expected_completed = t_wakeup + timedelta(seconds=2)
            expected_next = expected_completed + timedelta(seconds=60)
            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.ENRICHMENT_WORKER
                        )
                    )
                    is not None
                    and r.next_run_at == expected_next
                )
            )

            with session_factory() as session:
                enr = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
                enr_runs = session.scalars(
                    select(PipelineRun).where(
                        PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER
                    )
                ).all()

            assert enr is not None
            assert enr.status == PipelineLoopStatus.SLEEPING
            assert enr.last_started_at == t_wakeup
            assert enr.last_completed_at == expected_completed
            assert enr.last_completed_at >= enr.last_started_at
            assert enr.last_duration_ms == 2000
            assert enr.last_run_stats == {"enriched": 4}
            assert enr.last_error is None
            assert enr_runs == []
        finally:
            service.stop()
            run_thread.join(timeout=3.0)

    def test_run_unhandled_enrichment_error_and_queue_check_error_recover_on_next_wakeup(
        self, session_factory, monkeypatch
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        album_id = uuid.uuid4()
        with session_factory() as session, session.begin():
            album = Album(
                id=album_id,
                combined_artist="Recovery Artist",
                title="Recovery Album",
                fingerprint=f"fp-{album_id}",
                discogs_status="pending",
            )
            session.add(album)

        attempt = 0

        def flaky_process_pending(limit=None, shutdown_event=None):
            nonlocal attempt
            attempt += 1
            clock.advance(2)
            if attempt == 1:
                raise RuntimeError("enrichment worker crashed")
            with session_factory() as session, session.begin():
                row = session.get(Album, album_id)
                assert row is not None
                row.discogs_status = "matched"
            return 1

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []
        mock_enrichment = MagicMock()
        mock_enrichment.process_pending_enrichment.side_effect = flaky_process_pending

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=mock_enrichment,
            enrichment_interval_seconds=60,
            telemetry_recorder=recorder,
            clock=clock,
        )

        run_thread = threading.Thread(target=service.run, daemon=True)
        run_thread.start()
        try:
            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.TRACKING_POLL
                        )
                    )
                    is not None
                    and r.last_completed_at is not None
                )
            )

            # First wakeup: unhandled exception in enrichment service
            t_err_start = clock.advance(10)
            service.trigger_enrichment()

            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.ENRICHMENT_WORKER
                        )
                    )
                    is not None
                    and r.status == PipelineLoopStatus.ERROR
                )
            )

            with session_factory() as session:
                enr_err = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
                runs_err = session.scalars(
                    select(PipelineRun)
                    .where(PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER)
                    .order_by(PipelineRun.id.asc())
                ).all()

            assert enr_err is not None
            assert enr_err.last_started_at == t_err_start
            assert enr_err.last_completed_at == t_err_start + timedelta(seconds=2)
            assert enr_err.last_error == "enrichment worker crashed"
            assert enr_err.next_run_at == enr_err.last_completed_at + timedelta(seconds=60)
            assert len(runs_err) == 1
            assert runs_err[0].status == PipelineRunStatus.ERROR
            assert runs_err[0].error_message == "enrichment worker crashed"

            # Second wakeup: queue check raises unhandled DB error
            real_get_next = __import__(
                "reddit_scrape.service", fromlist=["get_next_queued_album"]
            ).get_next_queued_album
            monkeypatch.setattr(
                "reddit_scrape.service.get_next_queued_album",
                MagicMock(side_effect=RuntimeError("queue db timeout")),
            )
            clock.advance(60)
            service.trigger_enrichment()

            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.ENRICHMENT_WORKER
                        )
                    )
                    is not None
                    and r.last_error == "queue db timeout"
                )
            )

            # Third wakeup: restore queue query and recover with enriched=1
            monkeypatch.setattr(
                "reddit_scrape.service.get_next_queued_album",
                real_get_next,
            )
            clock.advance(60)
            service.trigger_enrichment()

            assert wait_until(
                lambda: (
                    (
                        r := get_loop_state(
                            session_factory, PipelineLoopName.ENRICHMENT_WORKER
                        )
                    )
                    is not None
                    and r.status == PipelineLoopStatus.SLEEPING
                )
            )

            with session_factory() as session:
                enr_rec = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
                runs_all = session.scalars(
                    select(PipelineRun)
                    .where(PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER)
                    .order_by(PipelineRun.id.asc())
                ).all()

            assert enr_rec is not None
            assert enr_rec.status == PipelineLoopStatus.SLEEPING
            assert enr_rec.last_error is None
            assert enr_rec.last_run_stats == {"enriched": 1}
            assert len(runs_all) == 3
            assert runs_all[0].status == PipelineRunStatus.ERROR
            assert runs_all[1].status == PipelineRunStatus.ERROR
            assert runs_all[1].error_message == "queue db timeout"
            assert runs_all[2].status == PipelineRunStatus.SUCCESS
            assert runs_all[2].stats == {"enriched": 1}
        finally:
            service.stop()
            run_thread.join(timeout=3.0)

    def test_run_once_enrichment_disabled_idle_active_and_error(self, session_factory):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        # 1. Disabled in run_once()
        disabled_service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=None,
            telemetry_recorder=recorder,
            clock=clock,
        )
        try:
            stats_disabled = disabled_service.run_once()
        finally:
            disabled_service.stop()

        assert stats_disabled.enriched_count == 0
        with session_factory() as session:
            enr_disabled = session.get(
                PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER
            )
            runs_disabled = session.scalars(
                select(PipelineRun).where(
                    PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER
                )
            ).all()
        assert enr_disabled is not None
        assert enr_disabled.status == PipelineLoopStatus.DISABLED
        assert enr_disabled.next_run_at is None
        assert runs_disabled == []

        # 2. Enabled idle in run_once() -> zero runs, next_run_at=None, stopped
        mock_enrichment = MagicMock()
        mock_enrichment.process_pending_enrichment.return_value = 0
        idle_service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=mock_enrichment,
            telemetry_recorder=recorder,
            clock=clock,
        )
        try:
            stats_idle = idle_service.run_once()
        finally:
            idle_service.stop()

        assert stats_idle.enriched_count == 0
        mock_enrichment.process_pending_enrichment.assert_not_called()
        with session_factory() as session:
            enr_idle = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
            runs_idle = session.scalars(
                select(PipelineRun).where(
                    PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER
                )
            ).all()
        assert enr_idle is not None
        assert enr_idle.status == PipelineLoopStatus.STOPPED
        assert enr_idle.next_run_at is None
        assert runs_idle == []

        # 3. Enabled active in run_once() -> logs success run with next_run_at=None
        album_id = uuid.uuid4()
        with session_factory() as session, session.begin():
            album = Album(
                id=album_id,
                combined_artist="Once Artist",
                title="Once Album",
                fingerprint=f"fp-{album_id}",
                discogs_status="pending",
            )
            session.add(album)

        observed_active_next_run: list[datetime | None] = []

        def fake_once_active(limit=None, shutdown_event=None):
            with session_factory() as session:
                state = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
                if state is not None:
                    observed_active_next_run.append(state.next_run_at)
            clock.advance(2)
            with session_factory() as session, session.begin():
                row = session.get(Album, album_id)
                assert row is not None
                row.discogs_status = "matched"
            return 3

        mock_enrichment.process_pending_enrichment.side_effect = fake_once_active
        active_service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=mock_enrichment,
            telemetry_recorder=recorder,
            clock=clock,
        )
        try:
            stats_active = active_service.run_once()
        finally:
            active_service.stop()

        assert stats_active.enriched_count == 3
        assert observed_active_next_run == [None]
        with session_factory() as session:
            enr_active = session.get(
                PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER
            )
            runs_active = session.scalars(
                select(PipelineRun)
                .where(PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER)
                .order_by(PipelineRun.id.asc())
            ).all()
        assert enr_active is not None
        assert enr_active.status == PipelineLoopStatus.STOPPED
        assert enr_active.next_run_at is None
        assert enr_active.last_run_stats == {"enriched": 3}
        assert len(runs_active) == 1
        assert runs_active[0].status == PipelineRunStatus.SUCCESS
        assert runs_active[0].stats == {"enriched": 3}

        # 4. Unhandled error in run_once() -> records error run, marks stopped, re-raises
        with session_factory() as session, session.begin():
            album_err = Album(
                combined_artist="Once Fail Artist",
                title="Once Fail Album",
                fingerprint=f"fp-{uuid.uuid4()}",
                discogs_status="pending",
            )
            session.add(album_err)

        def fake_once_fail(limit=None, shutdown_event=None):
            clock.advance(4)
            raise RuntimeError("once enrichment boom")

        mock_enrichment.process_pending_enrichment.side_effect = fake_once_fail
        fail_service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=mock_enrichment,
            telemetry_recorder=recorder,
            clock=clock,
        )
        try:
            with pytest.raises(RuntimeError, match="once enrichment boom"):
                fail_service.run_once()
        finally:
            fail_service.stop()

        with session_factory() as session:
            proc_fail = session.get(PipelineProcessStatus, "default")
            enr_fail = session.get(PipelineLoopState, PipelineLoopName.ENRICHMENT_WORKER)
            runs_all = session.scalars(
                select(PipelineRun)
                .where(PipelineRun.loop_name == PipelineLoopName.ENRICHMENT_WORKER)
                .order_by(PipelineRun.id.asc())
            ).all()
        assert proc_fail is not None
        assert proc_fail.status == PipelineProcessState.STOPPED
        assert proc_fail.run_mode == PipelineRunMode.ONCE
        assert enr_fail is not None
        assert enr_fail.status == PipelineLoopStatus.STOPPED
        assert enr_fail.next_run_at is None
        assert enr_fail.last_error == "once enrichment boom"
        assert len(runs_all) == 2
        assert runs_all[1].status == PipelineRunStatus.ERROR
        assert runs_all[1].error_message == "once enrichment boom"

    def test_enrichment_telemetry_db_failure_does_not_abort_enrichment(
        self, session_factory, caplog
    ):
        failing_factory = MagicMock(side_effect=RuntimeError("telemetry DB offline"))
        broken_recorder = PipelineTelemetryRecorder(
            failing_factory,
            clock=lambda: datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        with session_factory() as session, session.begin():
            session.add(
                Album(
                    combined_artist="Resilient Artist",
                    title="Resilient Album",
                    fingerprint=f"fp-{uuid.uuid4()}",
                    discogs_status="pending",
                )
            )

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []
        mock_enrichment = MagicMock()
        mock_enrichment.process_pending_enrichment.return_value = 1

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            enrichment_service=mock_enrichment,
            telemetry_recorder=broken_recorder,
        )
        try:
            with caplog.at_level(logging.WARNING, logger="reddit_scrape.telemetry"):
                stats = service.run_once()
        finally:
            service.stop()

        assert stats.enriched_count == 1
        mock_enrichment.process_pending_enrichment.assert_called_once()
        assert any("record_loop_start" in r.message for r in caplog.records)
        assert any("record_loop_completion" in r.message for r in caplog.records)


class TestEnrichmentTelemetryUnit:
    """Unit tests for refresh_loop_next_run failure isolation without live DB."""

    def test_refresh_loop_next_run_catches_db_errors_and_logs_warning(self, caplog):
        failing_factory = MagicMock(side_effect=RuntimeError("db connection lost"))
        recorder = PipelineTelemetryRecorder(
            failing_factory,
            clock=lambda: datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        with caplog.at_level(logging.WARNING, logger="reddit_scrape.telemetry"):
            recorder.refresh_loop_next_run(
                PipelineLoopName.ENRICHMENT_WORKER,
                next_run_at=datetime(2026, 6, 1, 12, 1, 0, tzinfo=timezone.utc),
                clear_error=True,
            )

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert "refresh_loop_next_run" in caplog.records[0].message
