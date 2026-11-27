"""Administrative snapshot retention and batched purging module."""

import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from reddit_scrape.db import get_session_factory

logger = logging.getLogger(__name__)

_PURGE_TARGETS: dict[str, tuple[str, str]] = {
    "submission_snapshots": ("captured_at", "snapshots"),
    "pipeline_runs": ("started_at", "pipeline runs"),
}


def _resolve_retention_days(*env_vars: str, default: int = 30) -> int:
    """Resolve an integer retention threshold across ordered environment variables."""
    load_dotenv()
    for var_name in env_vars:
        env_val = os.environ.get(var_name)
        if env_val:
            try:
                return int(env_val)
            except ValueError:
                pass
    return default


def get_default_retention_days() -> int:
    """Retrieve default retention days from SNAPSHOT_RETENTION_DAYS or 30."""
    return _resolve_retention_days("SNAPSHOT_RETENTION_DAYS", default=30)


def get_default_pipeline_run_retention_days() -> int:
    """Retrieve default pipeline run retention days from PIPELINE_RUN_RETENTION_DAYS,
    falling back to SNAPSHOT_RETENTION_DAYS, falling back to 30.
    """
    return _resolve_retention_days(
        "PIPELINE_RUN_RETENTION_DAYS",
        "SNAPSHOT_RETENTION_DAYS",
        default=30,
    )


def _execute_purge_loop(
    session: Session,
    retention_days: int,
    batch_size: int,
    cutoff: datetime | None = None,
    *,
    table_name: str = "submission_snapshots",
) -> int:
    """Execute batched deletion loop using PostgreSQL transaction per batch until 0 rows remain."""
    if table_name not in _PURGE_TARGETS:
        raise ValueError(f"Unsupported purge target table: {table_name}")
    timestamp_column, item_label = _PURGE_TARGETS[table_name]

    time_filter = ":cutoff" if cutoff is not None else "NOW() - (INTERVAL '1 day' * :retention_days)"
    stmt = text(
        f"""
        DELETE FROM {table_name}
        WHERE id IN (
            SELECT id FROM {table_name}
            WHERE {timestamp_column} < {time_filter}
            ORDER BY id ASC
            LIMIT :batch_size
        )
        """
    )
    params: dict[str, Any] = {"batch_size": batch_size}
    if cutoff is not None:
        params["cutoff"] = cutoff
    else:
        params["retention_days"] = retention_days

    total_deleted = 0
    while True:
        result = cast(CursorResult[Any], session.execute(stmt, params))
        session.commit()
        batch_deleted = int(result.rowcount or 0)
        total_deleted += batch_deleted
        logger.info(
            "Purged %d %s in batch (total deleted so far: %d)",
            batch_deleted,
            item_label,
            total_deleted,
        )
        if batch_deleted == 0:
            break

    return total_deleted


def _purge_table(
    table_name: str,
    session_factory_or_session: Callable[[], Session] | Session | None,
    retention_days: int,
    batch_size: int,
    cutoff: datetime | None,
) -> int:
    """Validate arguments and execute batched retention purge on a validated target table."""
    if retention_days < 0:
        raise ValueError("retention_days must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if session_factory_or_session is None:
        session_factory_or_session = get_session_factory()

    if isinstance(session_factory_or_session, Session):
        return _execute_purge_loop(
            session_factory_or_session,
            retention_days,
            batch_size,
            cutoff,
            table_name=table_name,
        )

    with session_factory_or_session() as session:
        return _execute_purge_loop(
            session,
            retention_days,
            batch_size,
            cutoff,
            table_name=table_name,
        )


def purge_snapshots(
    session_factory_or_session: Callable[[], Session] | Session | None = None,
    retention_days: int = 30,
    batch_size: int = 5000,
    *,
    cutoff: datetime | None = None,
) -> int:
    """Batched, chunked deletion of submission_snapshots older than retention_days.

    Preserves parent submissions and terminal engagement metrics permanently.
    Executes in transactions of up to batch_size until 0 rows remain.

    Args:
        session_factory_or_session: SQLAlchemy Session, session factory callable, or None (defaults to get_session_factory()).
        retention_days: Retention threshold in days (default 30).
        batch_size: Maximum rows deleted per batch transaction (default 5000).
        cutoff: Optional explicit cutoff datetime for testing/override.

    Returns:
        int: Total number of deleted submission_snapshots.
    """
    return _purge_table(
        "submission_snapshots",
        session_factory_or_session,
        retention_days,
        batch_size,
        cutoff,
    )


def purge_pipeline_runs(
    session_factory_or_session: Callable[[], Session] | Session | None = None,
    retention_days: int = 30,
    batch_size: int = 5000,
    *,
    cutoff: datetime | None = None,
) -> int:
    """Batched, chunked deletion of pipeline_runs older than retention_days.

    Preserves newer pipeline_runs as well as pipeline_process_status and
    pipeline_loop_states rows. Executes in transactions of up to batch_size
    until 0 rows remain.

    Args:
        session_factory_or_session: SQLAlchemy Session, session factory callable, or None (defaults to get_session_factory()).
        retention_days: Retention threshold in days (default 30).
        batch_size: Maximum rows deleted per batch transaction (default 5000).
        cutoff: Optional explicit cutoff datetime for testing/override.

    Returns:
        int: Total number of deleted pipeline_runs rows.
    """
    return _purge_table(
        "pipeline_runs",
        session_factory_or_session,
        retention_days,
        batch_size,
        cutoff,
    )
