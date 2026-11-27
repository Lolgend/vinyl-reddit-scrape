"""Isolated, defensively non-fatal pipeline operational telemetry persistence."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.sql import Executable

from reddit_scrape.models import (
    PipelineProcessState,
    PipelineProcessStatus,
    PipelineRunMode,
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

    def _execute_write(self, operation: str, stmt: Executable) -> None:
        """Execute a telemetry write in a short-lived isolated session."""
        try:
            with self.session_factory() as session, session.begin():
                session.execute(stmt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telemetry write failed during %s: %s", operation, exc)

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
        """Mark the Pipeline process stopped while preserving startup metadata."""
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
        self._upsert_process_status(
            "record_process_stop",
            insert_values,
            {
                "status": str(PipelineProcessState.STOPPED),
                "stopped_at": ts,
                "last_heartbeat_at": ts,
            },
        )


__all__ = ["PipelineTelemetryRecorder"]
