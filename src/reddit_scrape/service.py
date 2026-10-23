"""Continuous dual-loop ingestion poller service and pipeline runner."""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass
import inspect
import logging
import os
import signal
import threading
import time
from typing import Any, Coroutine, Mapping, Optional

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from reddit_scrape.db import get_database_url, get_session_factory
from reddit_scrape.extraction import ExtractionResult, ExtractionWorker
from reddit_scrape.ingestion import IngestionStats, RedditSourceAdapter, SubmissionIngester
from reddit_scrape.models import Submission
from reddit_scrape.persistence import ReleasePersistenceService

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Service configuration validated from environment variables."""

    reddit_client_id: str
    reddit_client_secret: str
    database_url: str
    deepseek_api_key: str
    poll_interval_seconds: int = 300
    hot_poll_interval_seconds: int = 900
    reddit_user_agent: str = "python:vinyl-releases-scraper:v0.1.0"
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"


def validate_environment(
    env: Optional[Mapping[str, str]] = None,
) -> ServiceConfig:
    """Validate required environment variables and return parsed ServiceConfig.

    Raises:
        ValueError: If any required configuration variable is missing.
    """
    if env is None:
        load_dotenv()
        env = os.environ

    missing = []
    reddit_client_id = env.get("REDDIT_CLIENT_ID")
    if not reddit_client_id or not str(reddit_client_id).strip():
        missing.append("REDDIT_CLIENT_ID")

    reddit_client_secret = env.get("REDDIT_CLIENT_SECRET")
    if not reddit_client_secret or not str(reddit_client_secret).strip():
        missing.append("REDDIT_CLIENT_SECRET")

    database_url = env.get("DATABASE_URL")
    if not database_url or not str(database_url).strip():
        missing.append("DATABASE_URL")

    deepseek_api_key = env.get("DEEPSEEK_API_KEY") or env.get("DEEPSEEK_KEY")
    if not deepseek_api_key or not str(deepseek_api_key).strip():
        missing.append("DEEPSEEK_API_KEY")

    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )

    # Normalize database URL
    normalized_db_url = get_database_url(str(database_url))

    def _parse_int(key: str, default: int) -> int:
        val = env.get(key)
        if val is None or not str(val).strip():
            return default
        try:
            return int(str(val).strip())
        except ValueError:
            return default

    poll_interval = _parse_int("POLL_INTERVAL_SECONDS", 300)
    hot_poll_interval = _parse_int("HOT_POLL_INTERVAL_SECONDS", 900)
    reddit_user_agent = env.get(
        "REDDIT_USER_AGENT", "python:vinyl-releases-scraper:v0.1.0"
    )
    deepseek_base_url = env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_model = env.get("DEEPSEEK_MODEL", "deepseek-chat")

    return ServiceConfig(
        reddit_client_id=str(reddit_client_id),
        reddit_client_secret=str(reddit_client_secret),
        database_url=normalized_db_url,
        deepseek_api_key=str(deepseek_api_key),
        poll_interval_seconds=poll_interval,
        hot_poll_interval_seconds=hot_poll_interval,
        reddit_user_agent=str(reddit_user_agent),
        deepseek_base_url=str(deepseek_base_url),
        deepseek_model=str(deepseek_model),
    )


@dataclass
class CycleStats:
    """Summary metrics returned by an ingestion and extraction cycle."""

    new_stats: IngestionStats
    hot_stats: IngestionStats
    extracted_count: int


class PollerService:
    """Continuous dual-loop Reddit scraper and entity extraction pipeline service."""

    def __init__(
        self,
        session_factory: Optional[sessionmaker[Session]] = None,
        reddit_adapter: Optional[RedditSourceAdapter] = None,
        extraction_worker: Optional[ExtractionWorker] = None,
        persistence_service: Optional[ReleasePersistenceService] = None,
        ingester: Optional[SubmissionIngester] = None,
        poll_interval_seconds: int = 300,
        hot_poll_interval_seconds: int = 900,
        new_limit: int = 50,
        hot_limit: int = 50,
    ) -> None:
        self.session_factory = session_factory or get_session_factory()
        self.reddit_adapter = reddit_adapter or RedditSourceAdapter()
        self.extraction_worker = extraction_worker or ExtractionWorker()
        self.persistence_service = persistence_service or ReleasePersistenceService(
            self.session_factory
        )
        self.ingester = ingester or SubmissionIngester(self.session_factory)

        self.poll_interval_seconds = poll_interval_seconds
        self.hot_poll_interval_seconds = hot_poll_interval_seconds
        self.new_limit = new_limit
        self.hot_limit = hot_limit

        self._shutdown_event = threading.Event()

    def stop(self) -> None:
        """Signal the poller service to initiate graceful termination."""
        self._shutdown_event.set()

    def is_shutdown_requested(self) -> bool:
        """Return True if termination signal has been received."""
        return self._shutdown_event.is_set()

    def register_signal_handlers(self) -> None:
        """Register graceful shutdown handlers for SIGINT and SIGTERM."""
        try:
            if threading.current_thread() is threading.main_thread():

                def _handler(sig: int, frame: Any) -> None:
                    logger.info(
                        "Received signal %s, initiating graceful shutdown...", sig
                    )
                    self.stop()

                signal.signal(signal.SIGINT, _handler)
                signal.signal(signal.SIGTERM, _handler)
        except (ValueError, AttributeError):
            pass

    def _run_coroutine(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Safely execute an asynchronous coroutine from synchronous context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(lambda: asyncio.run(coro)).result()
        else:
            return asyncio.run(coro)

    def _extract(self, title: str, flair: Optional[str]) -> ExtractionResult:
        """Invoke extraction worker, handling both sync and async return values."""
        res = self.extraction_worker.extract(title, flair=flair)
        if inspect.iscoroutine(res):
            return self._run_coroutine(res)
        return res

    def poll_new(self, limit: Optional[int] = None) -> IngestionStats:
        """Poll r/vinylreleases/new for recent release discovery."""
        fetch_limit = limit if limit is not None else self.new_limit
        if fetch_limit <= 0:
            return IngestionStats()
        items = self.reddit_adapter.fetch_new(limit=fetch_limit)
        return self.ingester.ingest_submissions(items)

    def poll_hot(self, limit: Optional[int] = None) -> IngestionStats:
        """Poll r/vinylreleases/hot for active engagement updates and flair mutation."""
        fetch_limit = limit if limit is not None else self.hot_limit
        if fetch_limit <= 0:
            return IngestionStats()
        items = self.reddit_adapter.fetch_hot(limit=fetch_limit)
        return self.ingester.ingest_submissions(items)

    def process_unextracted_submissions(
        self, batch_size: Optional[int] = None
    ) -> int:
        """Query and process unextracted submissions through extraction and persistence."""
        with self.session_factory() as session:
            stmt = (
                select(Submission.id, Submission.title, Submission.flair)
                .where(
                    Submission.is_extracted.is_(False),
                    Submission.is_release.is_(True),
                )
                .order_by(Submission.created_utc.asc())
            )
            if batch_size is not None:
                stmt = stmt.limit(batch_size)
            candidates = session.execute(stmt).all()

        if not candidates:
            return 0

        extracted_count = 0
        for sub_id, title, flair in candidates:
            if self._shutdown_event.is_set():
                logger.info("Shutdown requested, pausing extraction queue.")
                break

            try:
                extraction_result = self._extract(title, flair)
                self.persistence_service.persist(
                    submission_id=sub_id,
                    releases=extraction_result,
                )
                extracted_count += 1
            except Exception as exc:
                logger.error(
                    "Failed extraction or persistence for submission %s: %s",
                    sub_id,
                    exc,
                )

        return extracted_count

    def poll_and_extract_cycle(
        self,
        new_limit: Optional[int] = None,
        hot_limit: Optional[int] = None,
    ) -> CycleStats:
        """Execute a complete polling and extraction cycle."""
        n_limit = self.new_limit if new_limit is None else new_limit
        h_limit = self.hot_limit if hot_limit is None else hot_limit

        new_stats = self.poll_new(limit=n_limit)
        hot_stats = self.poll_hot(limit=h_limit)
        extracted_count = self.process_unextracted_submissions()

        return CycleStats(
            new_stats=new_stats,
            hot_stats=hot_stats,
            extracted_count=extracted_count,
        )

    def run_once(
        self,
        new_limit: Optional[int] = None,
        hot_limit: Optional[int] = None,
    ) -> CycleStats:
        """Run a single poll & extract cycle and return summary statistics."""
        return self.poll_and_extract_cycle(new_limit=new_limit, hot_limit=hot_limit)

    def run(self) -> None:
        """Start the continuous background ingestion poller and extraction loop."""
        self.register_signal_handlers()
        logger.info(
            "Starting PollerService (new_interval=%ss, hot_interval=%ss)",
            self.poll_interval_seconds,
            self.hot_poll_interval_seconds,
        )

        # Run initial cycle on startup
        try:
            logger.info("Executing initial discovery and tracking poll cycle...")
            self.poll_and_extract_cycle()
        except Exception as exc:
            logger.error("Error during initial poll and extract cycle: %s", exc)

        last_new_poll = time.time()
        last_hot_poll = time.time()

        try:
            while not self._shutdown_event.is_set():
                if self._shutdown_event.wait(timeout=1.0):
                    break

                now = time.time()
                did_poll = False

                if now - last_new_poll >= self.poll_interval_seconds:
                    try:
                        logger.info("Executing discovery poll (/new)...")
                        self.poll_new()
                        did_poll = True
                    except Exception as exc:
                        logger.error("Error during discovery poll (/new): %s", exc)
                    last_new_poll = time.time()

                if now - last_hot_poll >= self.hot_poll_interval_seconds:
                    try:
                        logger.info("Executing tracking poll (/hot)...")
                        self.poll_hot()
                        did_poll = True
                    except Exception as exc:
                        logger.error("Error during tracking poll (/hot): %s", exc)
                    last_hot_poll = time.time()

                if did_poll and not self._shutdown_event.is_set():
                    try:
                        extracted = self.process_unextracted_submissions()
                        if extracted > 0:
                            logger.info(
                                "Extracted and persisted %d release submissions",
                                extracted,
                            )
                    except Exception as exc:
                        logger.error(
                            "Error processing unextracted submissions: %s", exc
                        )

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in run loop.")
        finally:
            self.stop()
            logger.info("PollerService stopped cleanly.")


__all__ = [
    "CycleStats",
    "PollerService",
    "ServiceConfig",
    "validate_environment",
]
