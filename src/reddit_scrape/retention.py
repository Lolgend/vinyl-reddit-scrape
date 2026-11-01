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


def get_default_retention_days() -> int:
    """Retrieve default retention days from SNAPSHOT_RETENTION_DAYS or 30."""
    load_dotenv()
    env_val = os.environ.get("SNAPSHOT_RETENTION_DAYS")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            pass
    return 30


def _execute_purge_loop(
    session: Session,
    retention_days: int,
    batch_size: int,
    cutoff: datetime | None = None,
) -> int:
    """Execute batched deletion loop using PostgreSQL transaction per batch until 0 rows remain."""
    time_filter = ":cutoff" if cutoff is not None else "NOW() - (INTERVAL '1 day' * :retention_days)"
    stmt = text(
        f"""
        DELETE FROM submission_snapshots
        WHERE id IN (
            SELECT id FROM submission_snapshots
            WHERE captured_at < {time_filter}
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
            "Purged %d snapshots in batch (total deleted so far: %d)",
            batch_deleted,
            total_deleted,
        )
        if batch_deleted == 0:
            break

    return total_deleted


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
    if retention_days < 0:
        raise ValueError("retention_days must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    if session_factory_or_session is None:
        session_factory_or_session = get_session_factory()

    if isinstance(session_factory_or_session, Session):
        return _execute_purge_loop(
            session_factory_or_session, retention_days, batch_size, cutoff
        )

    with session_factory_or_session() as session:
        return _execute_purge_loop(session, retention_days, batch_size, cutoff)
