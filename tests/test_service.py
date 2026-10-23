import os
import signal
import threading
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

from reddit_scrape.cli import main
from reddit_scrape.db import get_engine, init_db
from reddit_scrape.extraction import ExtractedRelease, ExtractionResult
from reddit_scrape.ingestion import RawSubmission
from reddit_scrape.models import (
    Artist,
    Release,
    ReleaseArtist,
    Submission,
    SubmissionRelease,
    SubmissionSnapshot,
)
from reddit_scrape.service import (
    CycleStats,
    PollerService,
    ServiceConfig,
    validate_environment,
)


@pytest.fixture(scope="session")
def engine():
    try:
        eng = get_engine()
        init_db(eng)
        return eng
    except Exception as exc:
        pytest.skip(f"Database unavailable: {exc}")


@pytest.fixture(autouse=True)
def clean_db(engine):
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE submission_snapshots, submission_releases, "
                "release_artists, releases, artists, submissions CASCADE;"
            )
        )
    yield
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE submission_snapshots, submission_releases, "
                "release_artists, releases, artists, submissions CASCADE;"
            )
        )


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine)


class TestEnvironmentValidation:
    """Test environment validation and configuration parsing."""

    def test_validate_environment_success(self):
        env = {
            "REDDIT_CLIENT_ID": "mock_client_id",
            "REDDIT_CLIENT_SECRET": "mock_client_secret",
            "DATABASE_URL": "postgresql://test:test@localhost:5432/testdb",
            "DEEPSEEK_API_KEY": "mock_deepseek_key",
        }
        config = validate_environment(env)
        assert isinstance(config, ServiceConfig)
        assert config.reddit_client_id == "mock_client_id"
        assert config.reddit_client_secret == "mock_client_secret"
        assert "postgresql+psycopg://" in config.database_url
        assert config.deepseek_api_key == "mock_deepseek_key"
        assert config.poll_interval_seconds == 300
        assert config.hot_poll_interval_seconds == 900
        assert config.reddit_user_agent == "python:vinyl-releases-scraper:v0.1.0"

    def test_validate_environment_deepseek_key_alias(self):
        env = {
            "REDDIT_CLIENT_ID": "mock_client_id",
            "REDDIT_CLIENT_SECRET": "mock_client_secret",
            "DATABASE_URL": "postgresql://test:test@localhost:5432/testdb",
            "DEEPSEEK_KEY": "mock_deepseek_alias_key",
        }
        config = validate_environment(env)
        assert config.deepseek_api_key == "mock_deepseek_alias_key"

    def test_validate_environment_custom_intervals(self):
        env = {
            "REDDIT_CLIENT_ID": "mock_client_id",
            "REDDIT_CLIENT_SECRET": "mock_client_secret",
            "DATABASE_URL": "postgresql://test:test@localhost:5432/testdb",
            "DEEPSEEK_API_KEY": "mock_key",
            "POLL_INTERVAL_SECONDS": "60",
            "HOT_POLL_INTERVAL_SECONDS": "180",
        }
        config = validate_environment(env)
        assert config.poll_interval_seconds == 60
        assert config.hot_poll_interval_seconds == 180

    @pytest.mark.parametrize(
        "missing_key",
        [
            "REDDIT_CLIENT_ID",
            "REDDIT_CLIENT_SECRET",
            "DATABASE_URL",
            "DEEPSEEK_API_KEY",
        ],
    )
    def test_validate_environment_missing_keys(self, missing_key):
        env = {
            "REDDIT_CLIENT_ID": "mock_id",
            "REDDIT_CLIENT_SECRET": "mock_secret",
            "DATABASE_URL": "postgresql://test:test@localhost:5432/testdb",
            "DEEPSEEK_API_KEY": "mock_key",
        }
        del env[missing_key]
        with pytest.raises(ValueError) as excinfo:
            validate_environment(env)
        assert missing_key in str(excinfo.value)


class TestPollerServiceEndToEnd:
    """End-to-end integration tests for PollerService cycles."""

    def test_end_to_end_poll_and_extract_cycle(self, session_factory):
        created_now = datetime.now(timezone.utc)

        # Mock Reddit adapter with sample submissions:
        # 1. Single-artist release
        # 2. Non-release discussion
        # 3. Multi-artist release
        sub_single = RawSubmission(
            id="rel101",
            title="Radiohead - OK Computer (Blue Vinyl)",
            url="https://reddit.com/r/vinylreleases/rel101",
            permalink="/r/vinylreleases/comments/rel101",
            created_utc=created_now,
            score=10,
            upvote_ratio=0.95,
            num_comments=5,
            flair="New Release",
            author="collector_one",
            fullname="t3_rel101",
        )
        sub_discussion = RawSubmission(
            id="disc102",
            title="[Discussion] What is your favorite pressing ever?",
            url="https://reddit.com/r/vinylreleases/disc102",
            permalink="/r/vinylreleases/comments/disc102",
            created_utc=created_now,
            score=50,
            upvote_ratio=0.80,
            num_comments=100,
            flair="Discussion",
            author="curious_collector",
            fullname="t3_disc102",
        )
        sub_multi = RawSubmission(
            id="rel103",
            title="Converge / Dropdead - Split (7\" /500)",
            url="https://reddit.com/r/vinylreleases/rel103",
            permalink="/r/vinylreleases/comments/rel103",
            created_utc=created_now,
            score=35,
            upvote_ratio=0.99,
            num_comments=8,
            flair="Pre-Order",
            author="hardcore_fan",
            fullname="t3_rel103",
        )

        # Hot poll update: rel101 gets updated engagement and flair mutated to 'SOLD OUT'
        sub_hot_update = RawSubmission(
            id="rel101",
            title="Radiohead - OK Computer (Blue Vinyl)",
            url="https://reddit.com/r/vinylreleases/rel101",
            permalink="/r/vinylreleases/comments/rel101",
            created_utc=created_now,
            score=150,
            upvote_ratio=0.98,
            num_comments=42,
            flair="SOLD OUT",
            author="collector_one",
            fullname="t3_rel101",
        )

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = [sub_single, sub_discussion, sub_multi]
        mock_adapter.fetch_hot.return_value = [sub_hot_update]

        # Mock extraction worker
        mock_extraction_worker = MagicMock()

        def fake_extract(title, flair=None):
            if "Radiohead" in title:
                return ExtractionResult(
                    releases=[
                        ExtractedRelease(
                            artists=["Radiohead"],
                            combined_artist="Radiohead",
                            album_title="OK Computer",
                            variant="Blue Vinyl",
                            retailer=None,
                        )
                    ]
                )
            elif "Converge" in title:
                return ExtractionResult(
                    releases=[
                        ExtractedRelease(
                            artists=["Converge", "Dropdead"],
                            combined_artist="Converge / Dropdead",
                            album_title="Split",
                            variant='7" /500',
                            retailer="Deathwish",
                        )
                    ]
                )
            return ExtractionResult(releases=[])

        mock_extraction_worker.extract = AsyncMock(side_effect=fake_extract)

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=mock_extraction_worker,
        )

        stats = service.poll_and_extract_cycle(new_limit=50, hot_limit=50)

        assert isinstance(stats, CycleStats)
        assert stats.new_stats.inserted == 2
        assert stats.new_stats.skipped == 1
        assert stats.hot_stats.updated == 1
        assert stats.extracted_count == 2

        # Verify DB state
        with session_factory() as session:
            submissions = session.execute(
                select(Submission).order_by(Submission.id.asc())
            ).scalars().all()

            # Discussion skipped, only 2 releases inserted
            assert len(submissions) == 2
            sub_ids = [s.id for s in submissions]
            assert "t3_rel101" in sub_ids
            assert "t3_rel103" in sub_ids

            for sub in submissions:
                assert sub.is_extracted is True
                assert sub.is_release is True

            # Verify rel101 mutation to SOLD OUT and updated score
            sub101 = session.get(Submission, "t3_rel101")
            assert sub101.flair == "SOLD OUT"
            assert sub101.latest_score == 150
            assert sub101.latest_num_comments == 42

            # Verify snapshots (rel101 has 2 snapshots: 1 from /new, 1 from /hot)
            snaps_101 = session.execute(
                select(SubmissionSnapshot)
                .where(SubmissionSnapshot.submission_id == "t3_rel101")
                .order_by(SubmissionSnapshot.id.asc())
            ).scalars().all()
            assert len(snaps_101) == 2
            assert snaps_101[0].flair == "New Release"
            assert snaps_101[0].score == 10
            assert snaps_101[1].flair == "SOLD OUT"
            assert snaps_101[1].score == 150

            # Verify releases table
            releases = session.execute(
                select(Release).order_by(Release.album_title.asc())
            ).scalars().all()
            assert len(releases) == 2
            assert releases[0].album_title == "OK Computer"
            assert releases[0].combined_artist == "Radiohead"
            assert releases[0].variant == "Blue Vinyl"

            assert releases[1].album_title == "Split"
            assert releases[1].combined_artist == "Converge / Dropdead"
            assert releases[1].retailer_name == "Deathwish"

            # Verify artists table
            artists = session.execute(
                select(Artist).order_by(Artist.name.asc())
            ).scalars().all()
            artist_names = [a.name for a in artists]
            assert set(artist_names) == {"Radiohead", "Converge", "Dropdead"}

            # Verify release_artists join table
            split_rel = releases[1]
            ra_links = session.execute(
                select(ReleaseArtist).where(ReleaseArtist.release_id == split_rel.id)
            ).scalars().all()
            assert len(ra_links) == 2

            # Verify submission_releases join table
            sr_links = session.execute(
                select(SubmissionRelease).where(SubmissionRelease.submission_id == "t3_rel101")
            ).scalars().all()
            assert len(sr_links) == 1
            assert sr_links[0].release_id == releases[0].id

        # Re-running poll_and_extract_cycle when no new items exist extracts 0
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []
        stats2 = service.poll_and_extract_cycle(new_limit=50, hot_limit=50)
        assert stats2.extracted_count == 0

    def test_extraction_failure_does_not_corrupt_transactions(self, session_factory):
        created_now = datetime.now(timezone.utc)
        sub1 = RawSubmission(
            id="fail1",
            title="Failing Release - Crash LP",
            url="https://reddit.com/r/vinylreleases/fail1",
            permalink="/r/vinylreleases/comments/fail1",
            created_utc=created_now,
            fullname="t3_fail1",
        )
        sub2 = RawSubmission(
            id="ok2",
            title="Good Artist - Success EP",
            url="https://reddit.com/r/vinylreleases/ok2",
            permalink="/r/vinylreleases/comments/ok2",
            created_utc=created_now,
            fullname="t3_ok2",
        )

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = [sub1, sub2]
        mock_adapter.fetch_hot.return_value = []

        mock_worker = MagicMock()

        async def extract_with_failure(title, flair=None):
            if "Failing Release" in title:
                raise RuntimeError("LLM API connection timed out")
            return ExtractionResult(
                releases=[
                    ExtractedRelease(
                        artists=["Good Artist"],
                        combined_artist="Good Artist",
                        album_title="Success EP",
                    )
                ]
            )

        mock_worker.extract = AsyncMock(side_effect=extract_with_failure)

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=mock_worker,
        )

        stats = service.poll_and_extract_cycle()
        # sub2 was extracted, sub1 failed extraction
        assert stats.extracted_count == 1

        with session_factory() as session:
            failing = session.get(Submission, "t3_fail1")
            success = session.get(Submission, "t3_ok2")
            assert failing.is_extracted is False
            assert success.is_extracted is True

            # Releases has only Success EP
            releases = session.execute(select(Release)).scalars().all()
            assert len(releases) == 1
            assert releases[0].album_title == "Success EP"


class TestGracefulShutdown:
    """Test shutdown signaling and termination."""

    def test_service_stop_sets_event(self):
        service = PollerService(session_factory=MagicMock())
        assert not service.is_shutdown_requested()
        service.stop()
        assert service.is_shutdown_requested()

    def test_run_terminates_promptly_on_shutdown(self, session_factory):
        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = []
        mock_adapter.fetch_hot.return_value = []

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=MagicMock(),
            poll_interval_seconds=10,
            hot_poll_interval_seconds=10,
        )

        # Trigger stop in a background timer after 0.2s
        timer = threading.Timer(0.2, service.stop)
        timer.start()

        service.run()
        timer.join()

        assert service.is_shutdown_requested()

    def test_asynchronous_extraction_decoupled_from_polling(self, session_factory):
        """Verify LLM extraction executes in a decoupled background worker without blocking poller loop."""
        import time

        created_now = datetime.now(timezone.utc)
        sub = RawSubmission(
            id="async_rel_1",
            title="Slow Artist - Async LP",
            url="https://reddit.com/r/vinylreleases/async1",
            permalink="/r/vinylreleases/comments/async1",
            created_utc=created_now,
            fullname="t3_async_rel_1",
        )

        mock_adapter = MagicMock()
        mock_adapter.fetch_new.return_value = [sub]
        mock_adapter.fetch_hot.return_value = []

        mock_worker = MagicMock()

        # Simulate LLM API latency
        def slow_extract(title, flair=None):
            time.sleep(0.2)
            return ExtractionResult(
                releases=[
                    ExtractedRelease(
                        artists=["Slow Artist"],
                        combined_artist="Slow Artist",
                        album_title="Async LP",
                    )
                ]
            )

        mock_worker.extract = MagicMock(side_effect=slow_extract)

        service = PollerService(
            session_factory=session_factory,
            reddit_adapter=mock_adapter,
            extraction_worker=mock_worker,
            poll_interval_seconds=1,
            hot_poll_interval_seconds=1,
        )

        # Run service in background thread
        poller_thread = threading.Thread(target=service.run, daemon=True)
        poller_thread.start()

        # Wait briefly for initial poll and background extraction to trigger and finish
        time.sleep(0.6)
        service.stop()
        poller_thread.join(timeout=3.0)

        # Verify release was extracted by the background worker
        with session_factory() as session:
            persisted_sub = session.get(Submission, "t3_async_rel_1")
            assert persisted_sub is not None
            assert persisted_sub.is_extracted is True

            rel = session.execute(select(Release)).scalars().all()
            assert len(rel) == 1
            assert rel[0].album_title == "Async LP"


class TestCliRunner:
    """Test CLI subcommand `run` invocation and flags."""

    @patch("reddit_scrape.cli.validate_environment")
    @patch("reddit_scrape.cli.PollerService")
    @patch("reddit_scrape.cli.init_db")
    def test_cli_run_once(self, mock_init_db, mock_poller_cls, mock_validate_env):
        mock_validate_env.return_value = ServiceConfig(
            reddit_client_id="id",
            reddit_client_secret="sec",
            database_url="postgresql+psycopg://db",
            deepseek_api_key="key",
            poll_interval_seconds=300,
            hot_poll_interval_seconds=900,
        )
        mock_service = MagicMock()
        mock_service.run_once.return_value = CycleStats(
            new_stats=MagicMock(inserted=1, updated=0),
            hot_stats=MagicMock(inserted=0, updated=1),
            extracted_count=1,
        )
        mock_poller_cls.return_value = mock_service

        exit_code = main(["run", "--once"])
        assert exit_code == 0
        mock_validate_env.assert_called_once()
        mock_init_db.assert_called_once()
        mock_service.run_once.assert_called_once()

    @patch("reddit_scrape.cli.validate_environment")
    def test_cli_run_validation_failure(self, mock_validate_env):
        mock_validate_env.side_effect = ValueError("Missing REDDIT_CLIENT_ID")
        exit_code = main(["run", "--once"])
        assert exit_code == 1

    @patch("reddit_scrape.cli.validate_environment")
    @patch("reddit_scrape.cli.PollerService")
    @patch("reddit_scrape.cli.init_db")
    def test_cli_run_custom_intervals(self, mock_init_db, mock_poller_cls, mock_validate_env):
        mock_validate_env.return_value = ServiceConfig(
            reddit_client_id="id",
            reddit_client_secret="sec",
            database_url="postgresql+psycopg://db",
            deepseek_api_key="key",
            poll_interval_seconds=300,
            hot_poll_interval_seconds=900,
        )
        mock_service = MagicMock()
        mock_poller_cls.return_value = mock_service

        exit_code = main(["run", "--once", "--new-interval", "60", "--hot-interval", "180"])
        assert exit_code == 0
        mock_poller_cls.assert_called_once_with(
            poll_interval_seconds=60,
            hot_poll_interval_seconds=180,
            new_limit=50,
            hot_limit=50,
        )
