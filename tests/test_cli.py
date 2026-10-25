"""Tests for CLI arguments and subcommand execution."""

from unittest.mock import MagicMock, patch
import pytest

from reddit_scrape.cli import create_parser, main
from reddit_scrape.service import CycleStats, ServiceConfig


class TestCliThinkingLevel:
    """Tests for the --thinking-level CLI argument."""

    def test_parser_thinking_level_choices(self):
        parser = create_parser()
        # Parse run with valid thinking levels
        for level in ["none", "minimal", "low", "medium", "high", "max"]:
            args = parser.parse_args(["run", "--thinking-level", level])
            assert args.thinking_level == level

    def test_parser_thinking_level_invalid(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--thinking-level", "ultra"])

    @patch("reddit_scrape.cli.validate_environment")
    @patch("reddit_scrape.cli.PollerService")
    @patch("reddit_scrape.cli.init_db")
    def test_cli_run_passes_explicit_thinking_level(
        self, mock_init_db, mock_poller_cls, mock_validate_env
    ):
        mock_validate_env.return_value = ServiceConfig(
            reddit_client_id="id",
            reddit_client_secret="sec",
            database_url="postgresql+psycopg://db",
            deepseek_api_key="key",
            llm_thinking_level="low",
        )
        mock_service = MagicMock()
        mock_service.run_once.return_value = CycleStats(
            new_stats=MagicMock(inserted=0, updated=0),
            hot_stats=MagicMock(inserted=0, updated=0),
            extracted_count=0,
        )
        mock_poller_cls.return_value = mock_service

        exit_code = main(["run", "--once", "--thinking-level", "high"])
        assert exit_code == 0
        mock_poller_cls.assert_called_once_with(
            poll_interval_seconds=300,
            hot_poll_interval_seconds=900,
            new_limit=50,
            hot_limit=50,
            thinking_level="high",
        )

    @patch("reddit_scrape.cli.validate_environment")
    @patch("reddit_scrape.cli.PollerService")
    @patch("reddit_scrape.cli.init_db")
    def test_cli_run_defaults_thinking_level_from_config(
        self, mock_init_db, mock_poller_cls, mock_validate_env
    ):
        mock_validate_env.return_value = ServiceConfig(
            reddit_client_id="id",
            reddit_client_secret="sec",
            database_url="postgresql+psycopg://db",
            deepseek_api_key="key",
            llm_thinking_level="medium",
        )
        mock_service = MagicMock()
        mock_service.run_once.return_value = CycleStats(
            new_stats=MagicMock(inserted=0, updated=0),
            hot_stats=MagicMock(inserted=0, updated=0),
            extracted_count=0,
        )
        mock_poller_cls.return_value = mock_service

        exit_code = main(["run", "--once"])
        assert exit_code == 0
        mock_poller_cls.assert_called_once_with(
            poll_interval_seconds=300,
            hot_poll_interval_seconds=900,
            new_limit=50,
            hot_limit=50,
            thinking_level="medium",
        )
