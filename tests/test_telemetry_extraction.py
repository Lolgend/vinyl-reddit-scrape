"""PostgreSQL integration and unit tests for Extraction Queue telemetry and backoff."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

from reddit_scrape.db import get_engine, init_db
from reddit_scrape.extraction import ExtractedRelease, ExtractionResult
from reddit_scrape.models import (
    PipelineLoopName,
    PipelineLoopState,
    PipelineLoopStatus,
    PipelineRun,
    PipelineRunStatus,
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
    if request.cls and request.cls.__name__ == "TestExtractionTelemetryUnit":
        yield
        return
    engine = request.getfixturevalue("engine")
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


def _wait_until(predicate, *, timeout: float = 3.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _insert_submission(
    session_factory: sessionmaker[Session],
    sub_id: str,
    title: str,
    created_utc: datetime,
    *,
    is_extracted: bool = False,
    is_release: bool = True,
) -> None:
    reddit_id = sub_id.removeprefix("t3_")
    with session_factory() as session, session.begin():
        session.add(
            Submission(
                id=sub_id,
                reddit_id=reddit_id,
                title=title,
                url=f"https://reddit.com/r/vinylreleases/{reddit_id}",
                created_utc=created_utc,
                is_extracted=is_extracted,
                is_release=is_release,
            )
        )


class TestExtractionQueueTelemetryIntegration:
    """PostgreSQL integration tests for Extraction Queue telemetry, errors, and backoff."""

    def test_idle_wakeups_perform_zero_telemetry_writes_and_active_drain_records_run(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(
            session_factory, clock=clock, host="ext-host", pid=7001
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

        # Insert one non-release submission and one already-extracted submission:
        # neither is eligible for extraction.
        _insert_submission(
            session_factory,
            "t3_non_rel",
            "Question about sleeves",
            t0,
            is_release=False,
            is_extracted=False,
        )
        _insert_submission(
            session_factory,
            "t3_already_done",
            "Old Band - Old Album [LP]",
            t0,
            is_release=True,
            is_extracted=True,
        )

        observed_during_extract: list[tuple[str, datetime | None, datetime | None]] = []

        async def fake_extract_batch(batch_tuples, batch_size=5):
            with session_factory() as session:
                row = session.get(PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER)
                if row is not None:
                    observed_during_extract.append(
                        (row.status, row.last_started_at, row.next_run_at)
                    )
            clock.advance(2)
            return {
                sub_id: ExtractionResult(
                    releases=[
                        ExtractedRelease(
                            artists=[f"Artist {sub_id}"],
                            combined_artist=f"Artist {sub_id}",
                            album_title=title,
                        )
                    ]
                )
                for sub_id, title, _ in batch_tuples
            }

        mock_worker = MagicMock()
        mock_worker.extract_batch = AsyncMock(side_effect=fake_extract_batch)

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=mock_worker,
            poll_interval_seconds=600,
            hot_poll_interval_seconds=600,
            telemetry_recorder=recorder,
            clock=clock,
        )

        # Spy on telemetry recorder methods during idle wakeups
        with (
            patch.object(
                recorder, "record_loop_start", wraps=recorder.record_loop_start
            ) as spy_start,
            patch.object(
                recorder,
                "record_loop_completion",
                wraps=recorder.record_loop_completion,
            ) as spy_complete,
        ):
            run_thread = threading.Thread(target=service.run, daemon=True)
            run_thread.start()
            try:
                # Wait long enough for initial trigger + idle wakeup
                time.sleep(0.3)
                # Force a couple of idle extraction wakeups directly via trigger event
                service._extraction_trigger.set()
                time.sleep(0.15)
                service._extraction_trigger.set()
                time.sleep(0.15)

                ext_start_calls = [
                    c
                    for c in spy_start.call_args_list
                    if c.args and c.args[0] == PipelineLoopName.EXTRACTION_WORKER
                ]
                ext_complete_calls = [
                    c
                    for c in spy_complete.call_args_list
                    if c.args and c.args[0] == PipelineLoopName.EXTRACTION_WORKER
                ]
                assert ext_start_calls == []
                assert ext_complete_calls == []

                with session_factory() as session:
                    ext_idle = session.get(
                        PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER
                    )
                    ext_runs_idle = session.scalars(
                        select(PipelineRun).where(
                            PipelineRun.loop_name == PipelineLoopName.EXTRACTION_WORKER
                        )
                    ).all()
                assert ext_idle is not None
                assert ext_idle.status == PipelineLoopStatus.SLEEPING
                assert ext_idle.last_started_at is None
                assert ext_idle.last_completed_at is None
                assert ext_idle.next_run_at is None
                assert ext_runs_idle == []

                # Now insert two eligible submissions and trigger extraction
                t_active_start = clock.advance(5)
                _insert_submission(
                    session_factory,
                    "t3_ext_1",
                    "Active Artist One - Album One [LP]",
                    t_active_start,
                )
                _insert_submission(
                    session_factory,
                    "t3_ext_2",
                    "Active Artist Two - Album Two [LP]",
                    t_active_start,
                )
                service.trigger_extraction()

                assert _wait_until(
                    lambda: (
                        (
                            r := _get_loop_state(
                                session_factory, PipelineLoopName.EXTRACTION_WORKER
                            )
                        )
                        is not None
                        and r.last_completed_at is not None
                    )
                ), "Extraction worker did not record completion"

                assert observed_during_extract == [
                    (PipelineLoopStatus.RUNNING, t_active_start, None)
                ]

                with session_factory() as session:
                    ext_done = session.get(
                        PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER
                    )
                    ext_runs = session.scalars(
                        select(PipelineRun)
                        .where(
                            PipelineRun.loop_name == PipelineLoopName.EXTRACTION_WORKER
                        )
                        .order_by(PipelineRun.id.asc())
                    ).all()

                assert ext_done is not None
                assert ext_done.status == PipelineLoopStatus.SLEEPING
                assert ext_done.last_started_at == t_active_start
                assert ext_done.last_completed_at == t_active_start + timedelta(
                    seconds=2
                )
                assert ext_done.last_duration_ms == 2000
                assert ext_done.last_run_stats == {"extracted": 2}
                assert ext_done.last_error is None
                assert ext_done.next_run_at is None

                assert len(ext_runs) == 1
                assert ext_runs[0].status == PipelineRunStatus.SUCCESS
                assert ext_runs[0].stats == {"extracted": 2}
                assert ext_runs[0].duration_ms == 2000
                assert ext_runs[0].error_message is None
                assert ext_runs[0].host == "ext-host"
                assert ext_runs[0].pid == 7001
            finally:
                service.stop()
                run_thread.join(timeout=5.0)

    def test_partial_batch_errors_surface_without_backoff_and_clear_on_recovery(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        _insert_submission(session_factory, "t3_part_1", "Artist 1 - Album 1 [LP]", t0)
        _insert_submission(session_factory, "t3_part_2", "Artist 2 - Album 2 [LP]", t0)
        _insert_submission(session_factory, "t3_part_3", "Artist 3 - Album 3 [LP]", t0)

        attempt = 0

        async def fake_extract_batch(batch_tuples, batch_size=5):
            nonlocal attempt
            attempt += 1
            clock.advance(1)
            results: dict[str, ExtractionResult] = {}
            for sub_id, title, _ in batch_tuples:
                # On attempt 1, omit t3_part_2 so missing-result error triggers.
                if attempt == 1 and sub_id == "t3_part_2":
                    continue
                results[sub_id] = ExtractionResult(
                    releases=[
                        ExtractedRelease(
                            artists=[f"Artist {sub_id}"],
                            combined_artist=f"Artist {sub_id}",
                            album_title=title,
                        )
                    ]
                )
            return results

        mock_worker = MagicMock()
        mock_worker.extract_batch = AsyncMock(side_effect=fake_extract_batch)
        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=mock_worker,
            poll_interval_seconds=600,
            hot_poll_interval_seconds=600,
            telemetry_recorder=recorder,
            clock=clock,
        )

        orig_persist = service.persistence_service.persist
        persist_fail_once = True

        def flaky_persist(submission_id: str, releases: ExtractionResult) -> None:
            nonlocal persist_fail_once
            if persist_fail_once and submission_id == "t3_part_3":
                persist_fail_once = False
                raise RuntimeError("DB constraint boom")
            orig_persist(submission_id=submission_id, releases=releases)

        service.persistence_service.persist = flaky_persist  # type: ignore[method-assign]

        wait_timeouts: list[float] = []
        orig_wait = service._extraction_trigger.wait

        def spy_wait(timeout: float | None = None) -> bool:
            if timeout is not None:
                wait_timeouts.append(timeout)
            return orig_wait(timeout=timeout)

        with patch.object(service._extraction_trigger, "wait", side_effect=spy_wait):
            run_thread = threading.Thread(target=service.run, daemon=True)
            run_thread.start()
            try:
                assert _wait_until(
                    lambda: (
                        (
                            r := _get_loop_state(
                                session_factory, PipelineLoopName.EXTRACTION_WORKER
                            )
                        )
                        is not None
                        and r.status == PipelineLoopStatus.ERROR
                    )
                ), "Expected extraction_worker ERROR state after partial batch failures"

                with session_factory() as session:
                    err_state = session.get(
                        PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER
                    )
                    err_runs = session.scalars(
                        select(PipelineRun)
                        .where(
                            PipelineRun.loop_name == PipelineLoopName.EXTRACTION_WORKER
                        )
                        .order_by(PipelineRun.id.asc())
                    ).all()
                    s1 = session.get(Submission, "t3_part_1")
                    s2 = session.get(Submission, "t3_part_2")
                    s3 = session.get(Submission, "t3_part_3")

                assert s1 is not None and s1.is_extracted is True
                assert s2 is not None and s2.is_extracted is False
                assert s3 is not None and s3.is_extracted is False

                assert err_state is not None
                assert err_state.status == PipelineLoopStatus.ERROR
                assert err_state.last_run_stats == {"extracted": 1}
                assert err_state.next_run_at is None
                assert err_state.last_error is not None
                assert "2 error(s)" in err_state.last_error
                assert "No extraction result found for submission t3_part_2" in err_state.last_error

                assert len(err_runs) == 1
                assert err_runs[0].status == PipelineRunStatus.ERROR
                assert err_runs[0].stats == {"extracted": 1}
                assert err_runs[0].error_message == err_state.last_error

                # Trigger next drain; remaining 2 submissions succeed cleanly
                service.trigger_extraction()
                assert _wait_until(
                    lambda: (
                        (
                            r := _get_loop_state(
                                session_factory, PipelineLoopName.EXTRACTION_WORKER
                            )
                        )
                        is not None
                        and r.status == PipelineLoopStatus.SLEEPING
                        and r.last_run_stats == {"extracted": 2}
                    )
                ), "Expected extraction_worker recovery to SLEEPING"

                with session_factory() as session:
                    rec_state = session.get(
                        PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER
                    )
                    all_runs = session.scalars(
                        select(PipelineRun)
                        .where(
                            PipelineRun.loop_name == PipelineLoopName.EXTRACTION_WORKER
                        )
                        .order_by(PipelineRun.id.asc())
                    ).all()

                assert rec_state is not None
                assert rec_state.status == PipelineLoopStatus.SLEEPING
                assert rec_state.last_error is None
                assert rec_state.next_run_at is None
                assert rec_state.last_run_stats == {"extracted": 2}
                assert len(all_runs) == 2
                assert all_runs[1].status == PipelineRunStatus.SUCCESS
                assert all_runs[1].stats == {"extracted": 2}
                assert all_runs[1].error_message is None
                # Partial progress did not enter exponential backoff (>= 10s)
                assert all(t == 2.0 for t in wait_timeouts)
            finally:
                service.stop()
                run_thread.join(timeout=5.0)

    def test_zero_progress_backoff_progression_and_resets(self, session_factory):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        _insert_submission(session_factory, "t3_bo_1", "Backoff Artist - Album [LP]", t0)

        should_succeed = False
        fail_count = 0

        async def fake_extract_batch(batch_tuples, batch_size=5):
            nonlocal fail_count
            clock.advance(1)
            if not should_succeed:
                fail_count += 1
                raise RuntimeError(f"LLM outage #{fail_count}")
            return {
                sub_id: ExtractionResult(
                    releases=[
                        ExtractedRelease(
                            artists=["Backoff Artist"],
                            combined_artist="Backoff Artist",
                            album_title=title,
                        )
                    ]
                )
                for sub_id, title, _ in batch_tuples
            }

        mock_worker = MagicMock()
        mock_worker.extract_batch = AsyncMock(side_effect=fake_extract_batch)
        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=mock_worker,
            poll_interval_seconds=600,
            hot_poll_interval_seconds=600,
            telemetry_recorder=recorder,
            clock=clock,
        )

        # Direct backward-compatible call returns 0 without raising
        assert service.process_unextracted_submissions() == 0
        fail_count = 0

        recorded_waits: list[float] = []
        backoff_wait_entered = threading.Event()
        orig_wait = service._extraction_trigger.wait

        def spy_wait(timeout: float | None = None) -> bool:
            if timeout is not None:
                recorded_waits.append(timeout)
                if timeout >= 10.0:
                    backoff_wait_entered.set()
            return orig_wait(timeout=timeout)

        with patch.object(service._extraction_trigger, "wait", side_effect=spy_wait):
            run_thread = threading.Thread(target=service.run, daemon=True)
            run_thread.start()
            try:
                expected_backoffs = [10, 20, 40, 80, 160, 300, 300]
                for idx, expected_sec in enumerate(expected_backoffs, start=1):
                    assert _wait_until(
                        lambda target=idx: (
                            (
                                r := _get_loop_state(
                                    session_factory, PipelineLoopName.EXTRACTION_WORKER
                                )
                            )
                            is not None
                            and r.status == PipelineLoopStatus.ERROR
                            and fail_count >= target
                            and backoff_wait_entered.is_set()
                        )
                    ), f"Backoff iteration {idx} ({expected_sec}s) did not complete"

                    with session_factory() as session:
                        state = session.get(
                            PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER
                        )
                    assert state is not None
                    assert state.last_completed_at is not None
                    assert state.next_run_at == state.last_completed_at + timedelta(
                        seconds=expected_sec
                    )
                    assert state.last_run_stats == {"extracted": 0}
                    assert (
                        state.last_error
                        == f"Batch extraction failed for batch of 1 submissions: LLM outage #{idx}"
                    )

                    backoff_wait_entered.clear()
                    if idx < len(expected_backoffs):
                        # Wake internal wait directly without calling trigger_extraction()
                        service._extraction_trigger.set()

                backoff_waits = [t for t in recorded_waits if t >= 10.0]
                assert backoff_waits == [10.0, 20.0, 40.0, 80.0, 160.0, 300.0, 300.0]

                # trigger_extraction() resets backoff immediately (next failure starts at 10s again)
                backoff_wait_entered.clear()
                service.trigger_extraction()
                assert _wait_until(
                    lambda: fail_count >= 8 and backoff_wait_entered.is_set()
                ), "Triggered extraction did not run after backoff reset"

                with session_factory() as session:
                    reset_state = session.get(
                        PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER
                    )
                assert reset_state is not None
                assert reset_state.last_completed_at is not None
                assert reset_state.next_run_at == reset_state.last_completed_at + timedelta(
                    seconds=10
                )
                assert recorded_waits[-1] == 10.0

                # Succeeding extraction resets backoff back to 0 (normal 2.0s idle wait)
                should_succeed = True
                recorded_waits.clear()
                service._extraction_trigger.set()

                assert _wait_until(
                    lambda: (
                        (
                            r := _get_loop_state(
                                session_factory, PipelineLoopName.EXTRACTION_WORKER
                            )
                        )
                        is not None
                        and r.status == PipelineLoopStatus.SLEEPING
                        and r.last_run_stats == {"extracted": 1}
                    )
                ), "Extraction worker did not recover after backoff"

                with session_factory() as session:
                    ok_state = session.get(
                        PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER
                    )
                assert ok_state is not None
                assert ok_state.last_error is None
                assert ok_state.next_run_at is None
                assert _wait_until(lambda: 2.0 in recorded_waits)
            finally:
                service.stop()
                run_thread.join(timeout=5.0)

    def test_run_once_extraction_telemetry_idle_success_internal_error_and_unhandled(
        self, session_factory
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        clock = _FakeClock(t0)
        recorder = PipelineTelemetryRecorder(session_factory, clock=clock)

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []
        mock_worker = MagicMock()

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=mock_worker,
            telemetry_recorder=recorder,
            clock=clock,
        )

        # 1. Idle run_once (0 candidates) -> no extraction PipelineRun, no running flap
        with patch.object(
            recorder, "record_loop_start", wraps=recorder.record_loop_start
        ) as spy_start:
            stats_idle = service.run_once()
        assert stats_idle.extracted_count == 0
        ext_starts = [
            c
            for c in spy_start.call_args_list
            if c.args and c.args[0] == PipelineLoopName.EXTRACTION_WORKER
        ]
        assert ext_starts == []
        with session_factory() as session:
            ext_idle = session.get(PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER)
            ext_runs_idle = session.scalars(
                select(PipelineRun).where(
                    PipelineRun.loop_name == PipelineLoopName.EXTRACTION_WORKER
                )
            ).all()
        assert ext_idle is not None
        assert ext_idle.status == PipelineLoopStatus.STOPPED
        assert ext_idle.last_started_at is None
        assert ext_idle.next_run_at is None
        assert ext_runs_idle == []

        # 2. Active clean run_once (1 extracted) -> SUCCESS PipelineRun, next_run_at=None
        t_sub = clock.advance(5)
        _insert_submission(session_factory, "t3_once_ok", "Once Band - LP", t_sub)
        mock_worker.extract_batch.return_value = {
            "t3_once_ok": ExtractionResult(
                releases=[
                    ExtractedRelease(
                        artists=["Once Band"],
                        combined_artist="Once Band",
                        album_title="LP",
                    )
                ]
            )
        }
        stats_ok = service.run_once()
        assert stats_ok.extracted_count == 1
        with session_factory() as session:
            ext_ok = session.get(PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER)
            ext_runs_ok = session.scalars(
                select(PipelineRun)
                .where(PipelineRun.loop_name == PipelineLoopName.EXTRACTION_WORKER)
                .order_by(PipelineRun.id.asc())
            ).all()
        assert ext_ok is not None
        assert ext_ok.status == PipelineLoopStatus.STOPPED
        assert ext_ok.next_run_at is None
        assert ext_ok.last_run_stats == {"extracted": 1}
        assert ext_ok.last_error is None
        assert len(ext_runs_ok) == 1
        assert ext_runs_ok[0].status == PipelineRunStatus.SUCCESS
        assert ext_runs_ok[0].stats == {"extracted": 1}

        # 3. Internal batch failure in run_once -> ERROR PipelineRun, returns CycleStats
        t_err = clock.advance(5)
        _insert_submission(session_factory, "t3_once_err", "Err Band - LP", t_err)
        mock_worker.extract_batch.side_effect = RuntimeError("once batch fail")
        stats_err = service.run_once()
        assert stats_err.extracted_count == 0
        with session_factory() as session:
            ext_err = session.get(PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER)
            ext_runs_err = session.scalars(
                select(PipelineRun)
                .where(PipelineRun.loop_name == PipelineLoopName.EXTRACTION_WORKER)
                .order_by(PipelineRun.id.asc())
            ).all()
        assert ext_err is not None
        assert ext_err.status == PipelineLoopStatus.STOPPED
        assert ext_err.next_run_at is None
        assert ext_err.last_run_stats == {"extracted": 0}
        assert (
            ext_err.last_error
            == "Batch extraction failed for batch of 1 submissions: once batch fail"
        )
        assert len(ext_runs_err) == 2
        assert ext_runs_err[1].status == PipelineRunStatus.ERROR
        assert ext_runs_err[1].error_message == ext_err.last_error

        # 4. Unhandled exception during drain in run_once -> records ERROR, stops process/loops, re-raises
        def boom_drain(*args, **kwargs):
            clock.advance(2)
            raise RuntimeError("unhandled drain crash")

        service._drain_unextracted_submissions = boom_drain  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="unhandled drain crash"):
            service.run_once()

        with session_factory() as session:
            ext_boom = session.get(PipelineLoopState, PipelineLoopName.EXTRACTION_WORKER)
            ext_runs_boom = session.scalars(
                select(PipelineRun)
                .where(PipelineRun.loop_name == PipelineLoopName.EXTRACTION_WORKER)
                .order_by(PipelineRun.id.asc())
            ).all()
        assert ext_boom is not None
        assert ext_boom.status == PipelineLoopStatus.STOPPED
        assert ext_boom.next_run_at is None
        assert ext_boom.last_run_stats is None
        assert ext_boom.last_error == "unhandled drain crash"
        assert len(ext_runs_boom) == 3
        assert ext_runs_boom[2].status == PipelineRunStatus.ERROR
        assert ext_runs_boom[2].stats is None
        assert ext_runs_boom[2].error_message == "unhandled drain crash"
        service.stop()

    def test_extraction_telemetry_db_failure_does_not_abort_extraction(
        self, session_factory, caplog
    ):
        t0 = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        failing_factory = MagicMock(side_effect=RuntimeError("telemetry DB offline"))
        broken_recorder = PipelineTelemetryRecorder(
            failing_factory,
            clock=lambda: t0,
        )

        _insert_submission(
            session_factory,
            "t3_resilient_ext",
            "Resilient Ext Band - Album [LP]",
            t0,
        )

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []
        mock_worker = MagicMock()
        mock_worker.extract_batch.return_value = {
            "t3_resilient_ext": ExtractionResult(
                releases=[
                    ExtractedRelease(
                        artists=["Resilient Ext Band"],
                        combined_artist="Resilient Ext Band",
                        album_title="Album",
                    )
                ]
            )
        }

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=mock_worker,
            telemetry_recorder=broken_recorder,
        )
        try:
            with caplog.at_level(logging.WARNING, logger="reddit_scrape.telemetry"):
                stats = service.run_once()
        finally:
            service.stop()

        assert stats.extracted_count == 1
        with session_factory() as session:
            sub = session.get(Submission, "t3_resilient_ext")
        assert sub is not None
        assert sub.is_extracted is True
        assert any("record_loop_start" in r.message for r in caplog.records)
        assert any("record_loop_completion" in r.message for r in caplog.records)


class TestExtractionTelemetryUnit:
    """Unit tests for extraction error summary formatting and backoff helpers."""

    def test_error_summary_formatting_and_truncation(self):
        from reddit_scrape.service import _format_extraction_error_summary

        assert _format_extraction_error_summary(["single err"]) == "single err"
        assert (
            _format_extraction_error_summary(["first err", "second err"])
            == "2 error(s); first: first err"
        )
        long_err = "x" * 600
        truncated = _format_extraction_error_summary([long_err], max_length=100)
        assert len(truncated) == 100
        assert truncated.endswith("...")


def _get_loop_state(
    session_factory: sessionmaker[Session], loop_name: PipelineLoopName
) -> PipelineLoopState | None:
    with session_factory() as session:
        return session.get(PipelineLoopState, loop_name)
