"""Continuous dual-loop ingestion poller service and pipeline runner."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
import threading
import time
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from reddit_scrape.db import get_database_url, get_session_factory
from reddit_scrape.extraction import ExtractionResult, ExtractionWorker
from reddit_scrape.ingestion import (
    IngestionStats,
    RedditSourceAdapter,
    SubmissionIngester,
)
from reddit_scrape.models import Submission
from reddit_scrape.persistence import ReleasePersistenceService

logger = logging.getLogger(__name__)

VALID_THINKING_LEVELS = {"none", "minimal", "low", "medium", "high", "max"}


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
    llm_thinking_level: str = "low"
    llm_batch_size: int = 5


def validate_environment(
    env: Mapping[str, str] | None = None,
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

    raw_thinking_level = env.get("LLM_THINKING_LEVEL")
    if raw_thinking_level is None or not str(raw_thinking_level).strip():
        thinking_level = "low"
    else:
        thinking_level = str(raw_thinking_level).strip().lower()

    if thinking_level not in VALID_THINKING_LEVELS:
        raise ValueError(
            f"Invalid LLM_THINKING_LEVEL: {raw_thinking_level}. "
            f"Allowed: {', '.join(sorted(VALID_THINKING_LEVELS))}"
        )

    raw_batch_size = env.get("LLM_BATCH_SIZE")
    if raw_batch_size is None or not str(raw_batch_size).strip():
        llm_batch_size = 5
    else:
        try:
            llm_batch_size = int(str(raw_batch_size).strip())
        except ValueError:
            raise ValueError("LLM_BATCH_SIZE must be a positive integer")
        if llm_batch_size <= 0:
            raise ValueError("LLM_BATCH_SIZE must be a positive integer")

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
        llm_thinking_level=thinking_level,
        llm_batch_size=llm_batch_size,
    )


@dataclass
class CycleStats:
    """Summary metrics returned by an ingestion and extraction cycle."""

    new_stats: IngestionStats
    hot_stats: IngestionStats
    extracted_count: int


class _AsyncRunner:
    """Manages a persistent background event loop to execute coroutines safely across threads."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._run_loop,
                    name="PollerAsyncRunner",
                    daemon=True,
                )
                self._thread.start()

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        self._ensure_started()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def stop(self) -> None:
        with self._lock:
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=5.0)
            if self._loop is not None and not self._loop.is_closed():
                self._loop.close()
            self._loop = None
            self._thread = None


class PollerService:
    """Continuous dual-loop Reddit scraper and entity extraction pipeline service."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] | None = None,
        reddit_adapter: RedditSourceAdapter | None = None,
        extraction_worker: ExtractionWorker | None = None,
        persistence_service: ReleasePersistenceService | None = None,
        ingester: SubmissionIngester | None = None,
        poll_interval_seconds: int = 300,
        hot_poll_interval_seconds: int = 900,
        new_limit: int = 50,
        hot_limit: int = 50,
        thinking_level: str | None = None,
        llm_batch_size: int = 5,
    ) -> None:
        self.session_factory = session_factory or get_session_factory()
        self.reddit_adapter = reddit_adapter or RedditSourceAdapter()
        self.thinking_level = thinking_level
        self.llm_batch_size = llm_batch_size
        self.extraction_worker = extraction_worker or ExtractionWorker(
            thinking_level=thinking_level
        )
        self.persistence_service = persistence_service or ReleasePersistenceService(
            self.session_factory
        )
        self.ingester = ingester or SubmissionIngester(self.session_factory)

        self.poll_interval_seconds = poll_interval_seconds
        self.hot_poll_interval_seconds = hot_poll_interval_seconds
        self.new_limit = new_limit
        self.hot_limit = hot_limit

        self._shutdown_event = threading.Event()
        self._extraction_trigger = threading.Event()
        self._extraction_lock = threading.Lock()
        self._extraction_thread: threading.Thread | None = None
        self._async_runner = _AsyncRunner()

    def __del__(self) -> None:
        try:
            if hasattr(self, "_async_runner"):
                self._async_runner.stop()
        except Exception:
            pass

    def stop(self) -> None:
        """Signal the poller service to initiate graceful termination."""
        self._shutdown_event.set()
        self._extraction_trigger.set()
        if hasattr(self, "_async_runner"):
            self._async_runner.stop()

    def is_shutdown_requested(self) -> bool:
        """Return True if termination signal has been received."""
        return self._shutdown_event.is_set()

    def trigger_extraction(self) -> None:
        """Signal the background extraction worker that unextracted submissions are ready."""
        self._extraction_trigger.set()

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
        """Safely execute an asynchronous coroutine using the persistent event loop runner."""
        return self._async_runner.run(coro)

    def _extract(self, title: str, flair: str | None) -> ExtractionResult:
        """Invoke extraction worker, handling both sync and async return values."""
        res = self.extraction_worker.extract(title, flair=flair)
        if inspect.iscoroutine(res):
            return self._run_coroutine(res)
        return res

    def _extract_batch(
        self, chunk_tuples: list[tuple[str, str, str | None]]
    ) -> dict[str, ExtractionResult]:
        """Invoke extraction worker batch extraction, handling both sync and async return values."""
        if hasattr(self.extraction_worker, "extract_batch"):
            try:
                res = self.extraction_worker.extract_batch(chunk_tuples)
                if inspect.iscoroutine(res):
                    res = self._run_coroutine(res)
                if isinstance(res, dict):
                    return res
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "extract_batch failed (%s: %s); falling back to single extraction",
                    type(exc).__name__,
                    exc,
                )
        results: dict[str, ExtractionResult] = {}
        for sub_id, title, flair in chunk_tuples:
            try:
                results[sub_id] = self._extract(title, flair)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed extraction for submission %s: %s", sub_id, exc)
        return results

    def poll_new(self, limit: int | None = None) -> IngestionStats:
        """Poll r/vinylreleases/new for recent release discovery."""
        fetch_limit = limit if limit is not None else self.new_limit
        if fetch_limit <= 0:
            return IngestionStats()
        items = self.reddit_adapter.fetch_new(limit=fetch_limit)
        if items:
            logger.info("Discovered %d new submission(s)", len(items))
            for item in items:
                logger.info("  - [%s] %s", item.id, item.title)
        return self.ingester.ingest_submissions(items)

    def poll_hot(self, limit: int | None = None) -> IngestionStats:
        """Poll r/vinylreleases/hot for active engagement updates and flair mutation."""
        fetch_limit = limit if limit is not None else self.hot_limit
        if fetch_limit <= 0:
            return IngestionStats()
        items = self.reddit_adapter.fetch_hot(limit=fetch_limit)
        return self.ingester.ingest_submissions(items)

    def process_unextracted_submissions(
        self, batch_size: int | None = None
    ) -> int:
        """Query and process unextracted submissions through extraction and persistence."""
        with self._extraction_lock:
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

            chunk_size = max(1, self.llm_batch_size)
            chunks = [
                candidates[i : i + chunk_size]
                for i in range(0, len(candidates), chunk_size)
            ]

            extracted_count = 0
            for chunk in chunks:
                if self._shutdown_event.is_set():
                    logger.info("Shutdown requested, pausing extraction queue.")
                    break

                chunk_tuples = [(sub_id, title, flair) for sub_id, title, flair in chunk]
                start_time = time.perf_counter()
                try:
                    batch_results = self._extract_batch(chunk_tuples)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Batch extraction failed for chunk of %d submissions: %s",
                        len(chunk),
                        exc,
                    )
                    continue

                elapsed = time.perf_counter() - start_time
                logger.info(
                    "Batch extraction completed for %d submissions in %.2fs",
                    len(chunk),
                    elapsed,
                )

                for sub_id, title, flair in chunk:
                    try:
                        result = batch_results.get(sub_id)
                        if result is None:
                            logger.error(
                                "No extraction result found for submission %s",
                                sub_id,
                            )
                            continue
                        self.persistence_service.persist(
                            submission_id=sub_id,
                            releases=result,
                        )
                        extracted_count += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "Failed extraction or persistence for submission %s: %s",
                            sub_id,
                            exc,
                        )

            return extracted_count

    def poll_and_extract_cycle(
        self,
        new_limit: int | None = None,
        hot_limit: int | None = None,
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
        new_limit: int | None = None,
        hot_limit: int | None = None,
    ) -> CycleStats:
        """Run a single poll & extract cycle and return summary statistics."""
        return self.poll_and_extract_cycle(new_limit=new_limit, hot_limit=hot_limit)

    def _extraction_worker_loop(self) -> None:
        """Background extraction worker thread decoupled from ingestion polling."""
        logger.info("Extraction background worker thread started.")
        while not self._shutdown_event.is_set():
            # Wait for signal or periodic timeout to check unextracted queue
            self._extraction_trigger.wait(timeout=2.0)
            self._extraction_trigger.clear()
            if self._shutdown_event.is_set():
                break

            try:
                extracted = self.process_unextracted_submissions()
                if extracted > 0:
                    logger.info(
                        "Extracted and persisted %d release submissions in background",
                        extracted,
                    )
            except Exception as exc:
                logger.error(
                    "Error processing unextracted submissions in background: %s",
                    exc,
                )

    def run(self) -> None:
        """Start the continuous background ingestion poller with decoupled asynchronous extraction."""
        self.register_signal_handlers()
        logger.info(
            "Starting PollerService (new_interval=%ss, hot_interval=%ss)",
            self.poll_interval_seconds,
            self.hot_poll_interval_seconds,
        )

        # Start decoupled background extraction worker thread
        self._extraction_thread = threading.Thread(
            target=self._extraction_worker_loop,
            name="ExtractionWorker",
            daemon=True,
        )
        self._extraction_thread.start()

        # Run initial discovery and tracking polls on startup
        try:
            logger.info("Executing initial discovery and tracking poll cycle...")
            self.poll_new()
            self.poll_hot()
            self.trigger_extraction()
        except Exception as exc:
            logger.error("Error during initial poll cycle: %s", exc)

        last_new_poll = time.time()
        last_hot_poll = time.time()

        try:
            while not self._shutdown_event.is_set():
                if self._shutdown_event.wait(timeout=1.0):
                    break

                now = time.time()

                if now - last_new_poll >= self.poll_interval_seconds:
                    try:
                        logger.info("Executing discovery poll (/new)...")
                        self.poll_new()
                        self.trigger_extraction()
                    except Exception as exc:
                        logger.error("Error during discovery poll (/new): %s", exc)
                    last_new_poll = time.time()

                if now - last_hot_poll >= self.hot_poll_interval_seconds:
                    try:
                        logger.info("Executing tracking poll (/hot)...")
                        self.poll_hot()
                        self.trigger_extraction()
                    except Exception as exc:
                        logger.error("Error during tracking poll (/hot): %s", exc)
                    last_hot_poll = time.time()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in run loop.")
        finally:
            self.stop()
            if self._extraction_thread and self._extraction_thread.is_alive():
                self._extraction_thread.join(timeout=5.0)
            logger.info("PollerService stopped cleanly.")


__all__ = [
    "VALID_THINKING_LEVELS",
    "CycleStats",
    "PollerService",
    "ServiceConfig",
    "validate_environment",
]
