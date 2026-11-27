"""Isolated, defensively non-fatal pipeline operational telemetry persistence."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import Executable

from reddit_scrape.models import (
    PipelineLoopName,
    PipelineLoopState,
    PipelineLoopStatus,
    PipelineProcessState,
    PipelineProcessStatus,
    PipelineRun,
    PipelineRunMode,
    PipelineRunStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_PROCESS_NAME = "default"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PipelineTelemetryRecorder:
    """Persists Pipeline process liveness in isolated, non-fatal DB sessions."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        clock: Callable[[], datetime] | None = None,
        host: str | None = None,
        pid: int | None = None,
    ) -> None:
        self.session_factory = session_factory
        self._clock: Callable[[], datetime] = clock or _utc_now
        self.host = host if host is not None else socket.gethostname()
        self.pid = pid if pid is not None else os.getpid()

    def now(self) -> datetime:
        """Return the current timestamp from the configured clock."""
        return self._clock()

    def _run_write(self, operation: str, fn: Callable[[Session], Any]) -> None:
        """Execute a telemetry write callback in a short-lived isolated session."""
        try:
            with self.session_factory() as session, session.begin():
                fn(session)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telemetry write failed during %s: %s", operation, exc)

    def _execute_write(self, operation: str, stmt: Executable) -> None:
        """Execute a telemetry statement in a short-lived isolated session."""
        self._run_write(operation, lambda session: session.execute(stmt))

    def _upsert_process_status(
        self,
        operation: str,
        insert_values: dict[str, Any],
        conflict_set: dict[str, Any],
    ) -> None:
        """Upsert the singleton pipeline_process_status row on process_name."""
        stmt = (
            pg_insert(PipelineProcessStatus)
            .values(**insert_values)
            .on_conflict_do_update(
                index_elements=[PipelineProcessStatus.process_name],
                set_={**conflict_set, "updated_at": func.now()},
            )
        )
        self._execute_write(operation, stmt)

    def _upsert_loop_state(
        self,
        session: Session,
        insert_values: dict[str, Any],
        conflict_set: dict[str, Any],
    ) -> None:
        """Upsert a pipeline_loop_states row inside the caller's session."""
        stmt = (
            pg_insert(PipelineLoopState)
            .values(**insert_values)
            .on_conflict_do_update(
                index_elements=[PipelineLoopState.loop_name],
                set_={**conflict_set, "updated_at": func.now()},
            )
        )
        session.execute(stmt)

    def record_process_start(
        self,
        run_mode: PipelineRunMode | str,
        heartbeat_interval_seconds: int,
        started_at: datetime | None = None,
    ) -> None:
        """Record Pipeline startup in pipeline_process_status."""
        ts = started_at if started_at is not None else self.now()
        values: dict[str, Any] = {
            "process_name": DEFAULT_PROCESS_NAME,
            "status": str(PipelineProcessState.RUNNING),
            "run_mode": str(run_mode),
            "host": self.host,
            "pid": self.pid,
            "heartbeat_interval_seconds": heartbeat_interval_seconds,
            "started_at": ts,
            "last_heartbeat_at": ts,
            "stopped_at": None,
        }
        conflict_set = {
            key: value for key, value in values.items() if key != "process_name"
        }
        self._upsert_process_status("record_process_start", values, conflict_set)

    def record_heartbeat(self, heartbeat_at: datetime | None = None) -> None:
        """Refresh last_heartbeat_at while preserving existing process metadata."""
        ts = heartbeat_at if heartbeat_at is not None else self.now()
        insert_values: dict[str, Any] = {
            "process_name": DEFAULT_PROCESS_NAME,
            "status": str(PipelineProcessState.RUNNING),
            "run_mode": str(PipelineRunMode.CONTINUOUS),
            "host": self.host,
            "pid": self.pid,
            "heartbeat_interval_seconds": 30,
            "started_at": ts,
            "last_heartbeat_at": ts,
            "stopped_at": None,
        }
        self._upsert_process_status(
            "record_heartbeat",
            insert_values,
            {"last_heartbeat_at": ts},
        )

    def record_process_stop(self, stopped_at: datetime | None = None) -> None:
        """Mark process and non-disabled loops stopped while preserving run metrics."""
        ts = stopped_at if stopped_at is not None else self.now()
        insert_values: dict[str, Any] = {
            "process_name": DEFAULT_PROCESS_NAME,
            "status": str(PipelineProcessState.STOPPED),
            "run_mode": str(PipelineRunMode.CONTINUOUS),
            "host": self.host,
            "pid": self.pid,
            "heartbeat_interval_seconds": 30,
            "started_at": ts,
            "last_heartbeat_at": ts,
            "stopped_at": ts,
        }
        process_stmt = (
            pg_insert(PipelineProcessStatus)
            .values(**insert_values)
            .on_conflict_do_update(
                index_elements=[PipelineProcessStatus.process_name],
                set_={
                    "status": str(PipelineProcessState.STOPPED),
                    "stopped_at": ts,
                    "last_heartbeat_at": ts,
                    "updated_at": func.now(),
                },
            )
        )
        loops_stmt = (
            update(PipelineLoopState)
            .where(PipelineLoopState.status != str(PipelineLoopStatus.DISABLED))
            .values(
                status=str(PipelineLoopStatus.STOPPED),
                next_run_at=None,
                updated_at=func.now(),
            )
        )

        def _write(session: Session) -> None:
            session.execute(process_stmt)
            session.execute(loops_stmt)

        self._run_write("record_process_stop", _write)

    def seed_loop_states(
        self,
        intervals: Mapping[PipelineLoopName, int],
        *,
        enrichment_enabled: bool,
        is_continuous: bool,
        now: datetime | None = None,
    ) -> None:
        """Upsert all four Pipeline loop state rows on startup."""
        ts = now if now is not None else self.now()
        rows: list[dict[str, Any]] = []
        for loop_name in (
            PipelineLoopName.DISCOVERY_POLL,
            PipelineLoopName.TRACKING_POLL,
            PipelineLoopName.EXTRACTION_WORKER,
            PipelineLoopName.ENRICHMENT_WORKER,
        ):
            interval = intervals[loop_name]
            next_run_at: datetime | None = None
            if loop_name == PipelineLoopName.ENRICHMENT_WORKER:
                if not enrichment_enabled:
                    status = PipelineLoopStatus.DISABLED
                else:
                    status = PipelineLoopStatus.SLEEPING
                    if is_continuous:
                        next_run_at = ts + timedelta(seconds=interval)
            else:
                status = PipelineLoopStatus.SLEEPING
            rows.append(
                {
                    "loop_name": str(loop_name),
                    "status": str(status),
                    "interval_seconds": interval,
                    "next_run_at": next_run_at,
                }
            )

        stmt = pg_insert(PipelineLoopState).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=[PipelineLoopState.loop_name],
            set_={
                "status": stmt.excluded.status,
                "interval_seconds": stmt.excluded.interval_seconds,
                "next_run_at": stmt.excluded.next_run_at,
                "updated_at": func.now(),
            },
        )
        self._execute_write("seed_loop_states", stmt)

    def record_loop_start(
        self,
        loop_name: PipelineLoopName | str,
        *,
        started_at: datetime | None = None,
    ) -> None:
        """Mark a Pipeline loop as running and clear its next_run_at countdown."""
        ts = started_at if started_at is not None else self.now()
        insert_values: dict[str, Any] = {
            "loop_name": str(loop_name),
            "status": str(PipelineLoopStatus.RUNNING),
            "interval_seconds": 0,
            "last_started_at": ts,
            "next_run_at": None,
        }
        conflict_set = {
            "status": str(PipelineLoopStatus.RUNNING),
            "last_started_at": ts,
            "next_run_at": None,
        }
        self._run_write(
            "record_loop_start",
            lambda session: self._upsert_loop_state(session, insert_values, conflict_set),
        )

    def record_loop_completion(
        self,
        loop_name: PipelineLoopName | str,
        *,
        status: PipelineLoopStatus | str,
        started_at: datetime,
        completed_at: datetime,
        stats: dict[str, int] | None = None,
        error_message: str | None = None,
        next_run_at: datetime | None = None,
        record_run: bool = True,
    ) -> None:
        """Update loop completion metrics and optionally append a PipelineRun row."""
        duration_ms = max(0, int((completed_at - started_at).total_seconds() * 1000))
        status_str = str(status)
        last_error = (
            None if status_str == str(PipelineLoopStatus.SLEEPING) else error_message
        )
        loop_name_str = str(loop_name)

        state_values: dict[str, Any] = {
            "loop_name": loop_name_str,
            "status": status_str,
            "interval_seconds": 0,
            "last_started_at": started_at,
            "last_completed_at": completed_at,
            "next_run_at": next_run_at,
            "last_duration_ms": duration_ms,
            "last_run_stats": stats,
            "last_error": last_error,
        }
        conflict_set = {
            key: value for key, value in state_values.items() if key not in {"loop_name", "interval_seconds"}
        }

        run_status = (
            PipelineRunStatus.ERROR
            if status_str == str(PipelineLoopStatus.ERROR)
            else PipelineRunStatus.SUCCESS
        )
        run_error = error_message if run_status == PipelineRunStatus.ERROR else None

        def _write(session: Session) -> None:
            self._upsert_loop_state(session, state_values, conflict_set)
            if record_run:
                session.add(
                    PipelineRun(
                        loop_name=loop_name_str,
                        status=str(run_status),
                        started_at=started_at,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                        stats=stats,
                        error_message=run_error,
                        host=self.host,
                        pid=self.pid,
                    )
                )

        self._run_write("record_loop_completion", _write)

    def refresh_loop_next_run(
        self,
        loop_name: PipelineLoopName | str,
        *,
        next_run_at: datetime | None,
        clear_error: bool = True,
    ) -> None:
        """Refresh a loop's sleeping status and next_run_at without creating a PipelineRun."""
        loop_name_str = str(loop_name)
        insert_values: dict[str, Any] = {
            "loop_name": loop_name_str,
            "status": str(PipelineLoopStatus.SLEEPING),
            "interval_seconds": 0,
            "next_run_at": next_run_at,
        }
        conflict_set: dict[str, Any] = {
            "status": str(PipelineLoopStatus.SLEEPING),
            "next_run_at": next_run_at,
        }
        if clear_error:
            insert_values["last_error"] = None
            conflict_set["last_error"] = None
        self._run_write(
            "refresh_loop_next_run",
            lambda session: self._upsert_loop_state(
                session, insert_values, conflict_set
            ),
        )


__all__ = ["PipelineTelemetryRecorder"]
