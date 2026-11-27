"""Continuous dual-loop ingestion poller service and pipeline runner."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
import threading
import time
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from reddit_scrape.db import get_database_url, get_session_factory
from reddit_scrape.discogs import DiscogsClient
from reddit_scrape.enrichment import CatalogEnrichmentService, get_next_queued_album
from reddit_scrape.extraction import (
    ExtractionResult,
    ExtractionWorker,
    get_default_thinking_level,
)
from reddit_scrape.ingestion import (
    IngestionStats,
    RedditSourceAdapter,
    SubmissionIngester,
)
from reddit_scrape.models import (
    PipelineLoopName,
    PipelineLoopStatus,
    PipelineRunMode,
    Submission,
)
from reddit_scrape.persistence import ReleasePersistenceService
from reddit_scrape.telemetry import PipelineTelemetryRecorder

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
    llm_thinking_level: str = "none"
    llm_batch_size: int = 5
    discogs_token: str | None = None
    discogs_consumer_key: str | None = None
    discogs_consumer_secret: str | None = None
    discogs_user_agent: str = "VinylReleasesScraper/0.1.0"
    enrichment_interval_seconds: int = 60
    heartbeat_interval_seconds: int = 30


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
    if raw_thinking_level is not None and str(raw_thinking_level).strip():
        thinking_level = str(raw_thinking_level).strip().lower()
        if thinking_level not in VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid LLM_THINKING_LEVEL: {raw_thinking_level}. "
                f"Allowed: {', '.join(sorted(VALID_THINKING_LEVELS))}"
            )
    else:
        thinking_level = get_default_thinking_level(str(deepseek_model))

    def _parse_positive_int(key: str, default: int) -> int:
        raw_val = env.get(key)
        if raw_val is None or not str(raw_val).strip():
            return default
        try:
            parsed = int(str(raw_val).strip())
        except ValueError:
            raise ValueError(f"{key} must be a positive integer")
        if parsed <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return parsed

    llm_batch_size = _parse_positive_int("LLM_BATCH_SIZE", 5)

    # Discogs credentials validation
    raw_discogs_token = env.get("DISCOGS_TOKEN")
    discogs_token = (
        str(raw_discogs_token).strip()
        if raw_discogs_token and str(raw_discogs_token).strip()
        else None
    )

    raw_consumer_key = env.get("DISCOGS_CONSUMER_KEY")
    discogs_consumer_key = (
        str(raw_consumer_key).strip()
        if raw_consumer_key and str(raw_consumer_key).strip()
        else None
    )

    raw_consumer_secret = env.get("DISCOGS_CONSUMER_SECRET")
    discogs_consumer_secret = (
        str(raw_consumer_secret).strip()
        if raw_consumer_secret and str(raw_consumer_secret).strip()
        else None
    )

    has_token = bool(discogs_token)
    has_key_secret = bool(discogs_consumer_key and discogs_consumer_secret)

    if not (has_token or has_key_secret):
        raise ValueError(
            "Missing Discogs credentials: provide either DISCOGS_TOKEN "
            "or both DISCOGS_CONSUMER_KEY and DISCOGS_CONSUMER_SECRET"
        )

    raw_discogs_ua = env.get("DISCOGS_USER_AGENT")
    discogs_user_agent = (
        str(raw_discogs_ua).strip()
        if raw_discogs_ua and str(raw_discogs_ua).strip()
        else "VinylReleasesScraper/0.1.0"
    )

    enrichment_interval_seconds = _parse_positive_int(
        "ENRICHMENT_INTERVAL_SECONDS", 60
    )
    heartbeat_interval_seconds = _parse_positive_int(
        "PIPELINE_HEARTBEAT_INTERVAL_SECONDS", 30
    )

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
        discogs_token=discogs_token,
        discogs_consumer_key=discogs_consumer_key,
        discogs_consumer_secret=discogs_consumer_secret,
        discogs_user_agent=discogs_user_agent,
        enrichment_interval_seconds=enrichment_interval_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


@dataclass
class CycleStats:
    """Summary metrics returned by an ingestion and extraction cycle."""

    new_stats: IngestionStats
    hot_stats: IngestionStats
    extracted_count: int
    enriched_count: int = 0


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
            is_current_thread = (
                self._thread is not None and threading.current_thread() == self._thread
            )
            if self._loop is not None and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if not is_current_thread:
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
        enrichment_service: CatalogEnrichmentService | None = None,
        discogs_client: DiscogsClient | None = None,
        enrichment_interval_seconds: int = 60,
        discogs_token: str | None = None,
        discogs_consumer_key: str | None = None,
        discogs_consumer_secret: str | None = None,
        discogs_user_agent: str | None = None,
        deepseek_api_key: str | None = None,
        deepseek_base_url: str | None = None,
        heartbeat_interval_seconds: int = 30,
        telemetry_recorder: PipelineTelemetryRecorder | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.session_factory = session_factory or get_session_factory()
        self.reddit_adapter = reddit_adapter or RedditSourceAdapter()
        self.thinking_level = thinking_level
        self.llm_batch_size = llm_batch_size
        self.persistence_service = persistence_service or ReleasePersistenceService(
            self.session_factory
        )
        if extraction_worker is None:
            self.extraction_worker = ExtractionWorker(
                thinking_level=thinking_level,
                candidate_retriever=self.persistence_service.find_candidates,
            )
        else:
            self.extraction_worker = extraction_worker
        self.ingester = ingester or SubmissionIngester(self.session_factory)

        self.poll_interval_seconds = poll_interval_seconds
        self.hot_poll_interval_seconds = hot_poll_interval_seconds
        self.new_limit = new_limit
        self.hot_limit = hot_limit

        self.enrichment_interval_seconds = enrichment_interval_seconds
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self._clock = clock
        self.telemetry_recorder = telemetry_recorder or PipelineTelemetryRecorder(
            self.session_factory,
            clock=clock,
        )
        self._shutdown_event = threading.Event()
        self._extraction_trigger = threading.Event()
        self._extraction_lock = threading.Lock()
        self._extraction_thread: threading.Thread | None = None
        self._enrichment_trigger = threading.Event()
        self._enrichment_lock = threading.Lock()
        self._enrichment_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._async_runner = _AsyncRunner()
        self._discogs_client: DiscogsClient | None
        self.enrichment_service: CatalogEnrichmentService | None

        if enrichment_service is not None:
            self.enrichment_service = enrichment_service
            self._discogs_client = discogs_client or getattr(
                enrichment_service, "discogs_client", None
            )
        elif discogs_client is not None:
            self._discogs_client = discogs_client
            self.enrichment_service = CatalogEnrichmentService(
                session_factory=self.session_factory,
                discogs_client=discogs_client,
                deepseek_api_key=deepseek_api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY"),
                deepseek_base_url=deepseek_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
        elif discogs_token or (discogs_consumer_key and discogs_consumer_secret):
            raw_ua = discogs_user_agent
            ua = str(raw_ua).strip() if raw_ua and str(raw_ua).strip() else "VinylReleasesScraper/0.1.0"
            self._discogs_client = DiscogsClient(
                token=discogs_token,
                consumer_key=discogs_consumer_key,
                consumer_secret=discogs_consumer_secret,
                user_agent=ua,
                shutdown_event=self._shutdown_event,
            )
            self.enrichment_service = CatalogEnrichmentService(
                session_factory=self.session_factory,
                discogs_client=self._discogs_client,
                deepseek_api_key=deepseek_api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_KEY"),
                deepseek_base_url=deepseek_base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
        else:
            self._discogs_client = None
            self.enrichment_service = None

    def __del__(self) -> None:
        try:
            if hasattr(self, "_async_runner"):
                self._async_runner.stop()
        except Exception:  # noqa: BLE001, S110
            pass

    def stop(self) -> None:
        """Signal the poller service to initiate graceful termination."""
        self._shutdown_event.set()
        self._extraction_trigger.set()
        self._enrichment_trigger.set()
        if hasattr(self, "_async_runner"):
            self._async_runner.stop()

    def is_shutdown_requested(self) -> bool:
        """Return True if termination signal has been received."""
        return self._shutdown_event.is_set()

    def _now(self) -> datetime:
        """Return the current UTC timestamp from the injected clock or recorder."""
        if self._clock is not None:
            return self._clock()
        return self.telemetry_recorder.now()

    def _seed_loop_states(self, *, is_continuous: bool, now: datetime) -> None:
        """Seed all four Pipeline loop states with active intervals on startup."""
        self.telemetry_recorder.seed_loop_states(
            {
                PipelineLoopName.DISCOVERY_POLL: self.poll_interval_seconds,
                PipelineLoopName.TRACKING_POLL: self.hot_poll_interval_seconds,
                PipelineLoopName.EXTRACTION_WORKER: 2,
                PipelineLoopName.ENRICHMENT_WORKER: self.enrichment_interval_seconds,
            },
            enrichment_enabled=self.enrichment_service is not None,
            is_continuous=is_continuous,
            now=now,
        )

    def _execute_poll_with_telemetry(
        self,
        loop_name: PipelineLoopName,
        poll_fn: Callable[[], IngestionStats],
        *,
        interval_seconds: int,
        schedule_next: bool,
        raise_on_error: bool,
        error_log_message: str | None = None,
    ) -> tuple[datetime, IngestionStats | None]:
        """Execute a discovery or tracking poll with loop state and run telemetry."""
        started_at = self._now()
        self.telemetry_recorder.record_loop_start(loop_name, started_at=started_at)
        try:
            stats = poll_fn()
        except Exception as exc:
            completed_at = self._now()
            next_run_at = (
                completed_at + timedelta(seconds=interval_seconds)
                if schedule_next
                else None
            )
            self.telemetry_recorder.record_loop_completion(
                loop_name,
                status=PipelineLoopStatus.ERROR,
                started_at=started_at,
                completed_at=completed_at,
                stats=None,
                error_message=str(exc),
                next_run_at=next_run_at,
                record_run=True,
            )
            if raise_on_error:
                raise
            if error_log_message is not None:
                logger.error("%s: %s", error_log_message, exc)
            return completed_at, None

        completed_at = self._now()
        next_run_at = (
            completed_at + timedelta(seconds=interval_seconds)
            if schedule_next
            else None
        )
        run_stats = {
            "inserted": int(stats.inserted),
            "updated": int(stats.updated),
            "skipped": int(stats.skipped),
        }
        self.telemetry_recorder.record_loop_completion(
            loop_name,
            status=PipelineLoopStatus.SLEEPING,
            started_at=started_at,
            completed_at=completed_at,
            stats=run_stats,
            next_run_at=next_run_at,
            record_run=True,
        )
        return completed_at, stats

    def trigger_extraction(self) -> None:
        """Signal the background extraction worker that unextracted submissions are ready."""
        self._extraction_trigger.set()

    def trigger_enrichment(self) -> None:
        """Signal the background enrichment worker that pending albums are ready."""
        self._enrichment_trigger.set()

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
            return cast(ExtractionResult, self._run_coroutine(res))
        return cast(ExtractionResult, res)

    def _extract_batch(
        self, batch_tuples: list[tuple[str, str, str | None]]
    ) -> dict[str, ExtractionResult]:
        """Invoke extraction worker batch extraction, handling both sync and async return values."""
        if hasattr(self.extraction_worker, "extract_batch"):
            try:
                res = self.extraction_worker.extract_batch(
                    batch_tuples, batch_size=self.llm_batch_size
                )
                if inspect.iscoroutine(res):
                    res = self._run_coroutine(res)
            except TypeError as exc:
                if "batch_size" in str(exc):
                    res = self.extraction_worker.extract_batch(batch_tuples)
                    if inspect.iscoroutine(res):
                        res = self._run_coroutine(res)
                else:
                    raise
            if isinstance(res, dict):
                return res

        results: dict[str, ExtractionResult] = {}
        for sub_id, title, flair in batch_tuples:
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
        if items:
            logger.info("Polled /hot: tracked %d active submission(s)", len(items))
            for item in items:
                logger.debug(
                    "  - [%s] (score=%d, comments=%d) %s",
                    item.id,
                    item.score,
                    item.num_comments,
                    item.title,
                )
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

            partition_size = max(1, self.llm_batch_size)
            batches = [
                candidates[i : i + partition_size]
                for i in range(0, len(candidates), partition_size)
            ]

            extracted_count = 0
            for batch in batches:
                if self._shutdown_event.is_set():
                    logger.info("Shutdown requested, pausing extraction queue.")
                    break

                batch_tuples = [(sub_id, title, flair) for sub_id, title, flair in batch]
                start_time = time.perf_counter()
                try:
                    batch_results = self._extract_batch(batch_tuples)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Batch extraction failed for batch of %d submissions: %s",
                        len(batch),
                        exc,
                    )
                    continue

                elapsed = time.perf_counter() - start_time
                logger.info(
                    "Batch extraction completed for %d submissions in %.2fs",
                    len(batch),
                    elapsed,
                )

                for sub_id, title, flair in batch:
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

            if extracted_count > 0:
                self.trigger_enrichment()

            return extracted_count

    def process_pending_enrichment(self, limit: int | None = None) -> int:
        """Query and enrich pending/retryable albums using CatalogEnrichmentService."""
        if self.enrichment_service is None:
            return 0
        return self.enrichment_service.process_pending_enrichment(
            limit=limit,
            shutdown_event=self._shutdown_event,
        )

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
        if extracted_count > 0:
            self.trigger_enrichment()
        enriched_count = self.process_pending_enrichment()

        return CycleStats(
            new_stats=new_stats,
            hot_stats=hot_stats,
            extracted_count=extracted_count,
            enriched_count=enriched_count,
        )

    def run_once(
        self,
        new_limit: int | None = None,
        hot_limit: int | None = None,
    ) -> CycleStats:
        """Run a single poll & extract cycle and return summary statistics."""
        started_at = self._now()
        self.telemetry_recorder.record_process_start(
            run_mode=PipelineRunMode.ONCE,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
            started_at=started_at,
        )
        self._seed_loop_states(is_continuous=False, now=started_at)
        try:
            n_limit = self.new_limit if new_limit is None else new_limit
            h_limit = self.hot_limit if hot_limit is None else hot_limit

            _, new_stats = self._execute_poll_with_telemetry(
                PipelineLoopName.DISCOVERY_POLL,
                lambda: self.poll_new(limit=n_limit),
                interval_seconds=self.poll_interval_seconds,
                schedule_next=False,
                raise_on_error=True,
            )
            _, hot_stats = self._execute_poll_with_telemetry(
                PipelineLoopName.TRACKING_POLL,
                lambda: self.poll_hot(limit=h_limit),
                interval_seconds=self.hot_poll_interval_seconds,
                schedule_next=False,
                raise_on_error=True,
            )
            assert new_stats is not None
            assert hot_stats is not None

            extracted_count = self.process_unextracted_submissions()
            if extracted_count > 0:
                self.trigger_enrichment()
            enriched_count = (
                self._execute_enrichment_with_telemetry(
                    schedule_next=False,
                    raise_on_error=True,
                )
                if self.enrichment_service is not None
                else 0
            )

            return CycleStats(
                new_stats=new_stats,
                hot_stats=hot_stats,
                extracted_count=extracted_count,
                enriched_count=enriched_count,
            )
        finally:
            self.telemetry_recorder.record_process_stop(stopped_at=self._now())

    def _heartbeat_loop(self) -> None:
        """Background heartbeat thread refreshing pipeline process liveness."""
        logger.debug("Pipeline heartbeat thread started.")
        while not self._shutdown_event.wait(
            timeout=float(self.heartbeat_interval_seconds)
        ):
            try:
                self.telemetry_recorder.record_heartbeat(heartbeat_at=self._now())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unexpected error in pipeline heartbeat loop: %s", exc)

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
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Error processing unextracted submissions in background: %s",
                    exc,
                )

    def _enrichment_next_run_at(
        self, base: datetime, schedule_next: bool
    ) -> datetime | None:
        """Compute the next enrichment run timestamp when scheduling is enabled."""
        if not schedule_next:
            return None
        return base + timedelta(seconds=self.enrichment_interval_seconds)

    def _record_enrichment_error(
        self,
        started_at: datetime,
        exc: Exception,
        *,
        schedule_next: bool,
        log_prefix: str,
    ) -> None:
        """Record an unhandled enrichment error on loop state and run history."""
        completed_at = self._now()
        self.telemetry_recorder.record_loop_completion(
            PipelineLoopName.ENRICHMENT_WORKER,
            status=PipelineLoopStatus.ERROR,
            started_at=started_at,
            completed_at=completed_at,
            stats=None,
            error_message=str(exc),
            next_run_at=self._enrichment_next_run_at(completed_at, schedule_next),
            record_run=True,
        )
        logger.error("%s: %s", log_prefix, exc)

    def _execute_enrichment_with_telemetry(
        self,
        *,
        schedule_next: bool,
        raise_on_error: bool,
    ) -> int:
        """Check enrichment queue and run pending enrichment with loop telemetry."""
        check_started_at = self._now()
        try:
            with self.session_factory() as session:
                has_work = (
                    get_next_queued_album(session, now=check_started_at) is not None
                )
        except Exception as exc:
            self._record_enrichment_error(
                check_started_at,
                exc,
                schedule_next=schedule_next,
                log_prefix="Error checking enrichment queue",
            )
            if raise_on_error:
                raise
            return 0

        if not has_work:
            now_ts = self._now()
            self.telemetry_recorder.refresh_loop_next_run(
                PipelineLoopName.ENRICHMENT_WORKER,
                next_run_at=self._enrichment_next_run_at(now_ts, schedule_next),
                clear_error=True,
            )
            return 0

        started_at = self._now()
        self.telemetry_recorder.record_loop_start(
            PipelineLoopName.ENRICHMENT_WORKER,
            started_at=started_at,
        )
        try:
            enriched = self.process_pending_enrichment()
        except Exception as exc:
            self._record_enrichment_error(
                started_at,
                exc,
                schedule_next=schedule_next,
                log_prefix="Error processing pending enrichment in background",
            )
            if raise_on_error:
                raise
            return 0

        completed_at = self._now()
        next_run_at = self._enrichment_next_run_at(completed_at, schedule_next)
        if enriched > 0:
            self.telemetry_recorder.record_loop_completion(
                PipelineLoopName.ENRICHMENT_WORKER,
                status=PipelineLoopStatus.SLEEPING,
                started_at=started_at,
                completed_at=completed_at,
                stats={"enriched": int(enriched)},
                next_run_at=next_run_at,
                record_run=True,
            )
            logger.info(
                "Enriched %d album(s) with Discogs catalog in background",
                enriched,
            )
        else:
            self.telemetry_recorder.refresh_loop_next_run(
                PipelineLoopName.ENRICHMENT_WORKER,
                next_run_at=next_run_at,
                clear_error=True,
            )
        return int(enriched)

    def _enrichment_worker_loop(self) -> None:
        """Background Discogs enrichment worker thread decoupled from ingestion and extraction."""
        logger.info("Discogs enrichment background worker thread started.")
        while not self._shutdown_event.is_set():
            self._enrichment_trigger.wait(timeout=float(self.enrichment_interval_seconds))
            self._enrichment_trigger.clear()
            if self._shutdown_event.is_set():
                break

            self._execute_enrichment_with_telemetry(
                schedule_next=True,
                raise_on_error=False,
            )

    def run(self) -> None:
        """Start the continuous background ingestion poller with decoupled asynchronous extraction."""
        self.register_signal_handlers()
        logger.info(
            "Starting PollerService (new_interval=%ss, hot_interval=%ss, enrichment_interval=%ss, heartbeat_interval=%ss)",
            self.poll_interval_seconds,
            self.hot_poll_interval_seconds,
            self.enrichment_interval_seconds,
            self.heartbeat_interval_seconds,
        )

        started_at = self._now()
        self.telemetry_recorder.record_process_start(
            run_mode=PipelineRunMode.CONTINUOUS,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
            started_at=started_at,
        )
        self._seed_loop_states(is_continuous=True, now=started_at)

        try:
            # Start dedicated background pipeline heartbeat thread
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="PipelineHeartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

            # Start decoupled background extraction worker thread
            self._extraction_thread = threading.Thread(
                target=self._extraction_worker_loop,
                name="ExtractionWorker",
                daemon=True,
            )
            self._extraction_thread.start()

            # Start decoupled background Discogs enrichment worker thread when enabled
            if self.enrichment_service is not None:
                self._enrichment_thread = threading.Thread(
                    target=self._enrichment_worker_loop,
                    name="DiscogsEnrichmentWorker",
                    daemon=True,
                )
                self._enrichment_thread.start()

            # Run initial discovery and tracking polls on startup
            logger.info("Executing initial discovery and tracking poll cycle...")
            last_new_poll, new_stats = self._execute_poll_with_telemetry(
                PipelineLoopName.DISCOVERY_POLL,
                self.poll_new,
                interval_seconds=self.poll_interval_seconds,
                schedule_next=True,
                raise_on_error=False,
                error_log_message="Error during initial discovery poll (/new)",
            )
            last_hot_poll, hot_stats = self._execute_poll_with_telemetry(
                PipelineLoopName.TRACKING_POLL,
                self.poll_hot,
                interval_seconds=self.hot_poll_interval_seconds,
                schedule_next=True,
                raise_on_error=False,
                error_log_message="Error during initial tracking poll (/hot)",
            )
            if new_stats is not None or hot_stats is not None:
                self.trigger_extraction()
            logger.info(
                "Initial poll cycle completed. Poller active (next discovery poll in %ds, tracking poll in %ds).",
                self.poll_interval_seconds,
                self.hot_poll_interval_seconds,
            )

            while not self._shutdown_event.is_set():
                if self._shutdown_event.wait(timeout=1.0):
                    break

                now = self._now()

                if now >= last_new_poll + timedelta(seconds=self.poll_interval_seconds):
                    logger.info("Executing discovery poll (/new)...")
                    last_new_poll, new_stats = self._execute_poll_with_telemetry(
                        PipelineLoopName.DISCOVERY_POLL,
                        self.poll_new,
                        interval_seconds=self.poll_interval_seconds,
                        schedule_next=True,
                        raise_on_error=False,
                        error_log_message="Error during discovery poll (/new)",
                    )
                    if new_stats is not None:
                        self.trigger_extraction()

                if now >= last_hot_poll + timedelta(seconds=self.hot_poll_interval_seconds):
                    logger.info("Executing tracking poll (/hot)...")
                    last_hot_poll, hot_stats = self._execute_poll_with_telemetry(
                        PipelineLoopName.TRACKING_POLL,
                        self.poll_hot,
                        interval_seconds=self.hot_poll_interval_seconds,
                        schedule_next=True,
                        raise_on_error=False,
                        error_log_message="Error during tracking poll (/hot)",
                    )
                    if hot_stats is not None:
                        self.trigger_extraction()

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in run loop.")
        finally:
            self.stop()
            if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                self._heartbeat_thread.join(timeout=5.0)
            if self._extraction_thread and self._extraction_thread.is_alive():
                self._extraction_thread.join(timeout=5.0)
            if self._enrichment_thread and self._enrichment_thread.is_alive():
                self._enrichment_thread.join(timeout=5.0)
            self.telemetry_recorder.record_process_stop(stopped_at=self._now())
            logger.info("PollerService stopped cleanly.")


__all__ = [
    "VALID_THINKING_LEVELS",
    "CycleStats",
    "PollerService",
    "ServiceConfig",
    "validate_environment",
]
