"""Administrative snapshot retention and batched purging module."""

from datetime import datetime, timezone
import logging
import os
from typing import Any, Optional, Union
from dotenv import load_dotenv
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session, sessionmaker

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
    cutoff: Optional[datetime] = None,
) -> int:
    """Execute batched deletion loop using transaction per batch until 0 rows remain."""
    bind = session.get_bind()
    dialect_name = bind.dialect.name if bind else "postgresql"

    if cutoff is not None:
        stmt = text(
            """
            DELETE FROM submission_snapshots
            WHERE id IN (
                SELECT id FROM submission_snapshots
                WHERE captured_at < :cutoff
                ORDER BY id ASC
                LIMIT :batch_size
            )
            """
        )
        params: dict[str, Any] = {"cutoff": cutoff, "batch_size": batch_size}
    elif dialect_name == "postgresql":
        stmt = text(
            """
            DELETE FROM submission_snapshots
            WHERE id IN (
                SELECT id FROM submission_snapshots
                WHERE captured_at < NOW() - (INTERVAL '1 day' * :retention_days)
                ORDER BY id ASC
                LIMIT :batch_size
            )
            """
        )
        params = {"retention_days": retention_days, "batch_size": batch_size}
    else:
        # Fallback for non-postgres dialects (e.g. sqlite)
        from datetime import timedelta

        computed_cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        stmt = text(
            """
            DELETE FROM submission_snapshots
            WHERE id IN (
                SELECT id FROM submission_snapshots
                WHERE captured_at < :cutoff
                ORDER BY id ASC
                LIMIT :batch_size
            )
            """
        )
        params = {"cutoff": computed_cutoff, "batch_size": batch_size}

    total_deleted = 0
    while True:
        result = session.execute(stmt, params)
        session.commit()
        batch_deleted = result.rowcount
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
    session_factory_or_session: Union[sessionmaker, Session, Engine, Any] = None,
    retention_days: int = 30,
    batch_size: int = 5000,
    *,
    cutoff: Optional[datetime] = None,
) -> int:
    """Batched, chunked deletion of submission_snapshots older than retention_days.

    Preserves parent submissions and terminal engagement metrics permanently.
    Executes in transactions of up to batch_size until 0 rows remain.

    Args:
        session_factory_or_session: SQLAlchemy Session, sessionmaker, Engine, or None (defaults to get_session_factory()).
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
    elif isinstance(session_factory_or_session, Engine):
        factory = sessionmaker(bind=session_factory_or_session)
        with factory() as session:
            return _execute_purge_loop(session, retention_days, batch_size, cutoff)
    elif callable(session_factory_or_session):
        with session_factory_or_session() as session:
            return _execute_purge_loop(session, retention_days, batch_size, cutoff)
    else:
        raise TypeError(
            f"Unsupported session_factory_or_session type: {type(session_factory_or_session)}"
        )
