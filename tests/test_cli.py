"""Tests for CLI arguments and subcommand execution."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from reddit_scrape.cli import configure_logging, create_parser, main
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
            llm_batch_size=5,
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
            llm_batch_size=5,
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
            llm_batch_size=5,
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
            llm_batch_size=5,
        )


class TestCliBatchSizeAndVerbosity:
    """Tests for verbose logging, log levels, and batch size options."""

    def test_parser_verbose_flags(self):
        parser = create_parser()
        args_short = parser.parse_args(["run", "-v"])
        assert args_short.verbose is True

        args_long = parser.parse_args(["run", "--verbose"])
        assert args_long.verbose is True

        args_default = parser.parse_args(["run"])
        assert args_default.verbose is False

    def test_parser_log_level_choices(self):
        parser = create_parser()
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            args = parser.parse_args(["run", "--log-level", level])
            assert args.log_level == level

    def test_parser_log_level_invalid(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--log-level", "TRACE"])

    def test_parser_llm_batch_size(self):
        parser = create_parser()
        args = parser.parse_args(["run", "--llm-batch-size", "10"])
        assert args.llm_batch_size == 10

        args_default = parser.parse_args(["run"])
        assert args_default.llm_batch_size is None

    @patch("reddit_scrape.cli.logging.basicConfig")
    def test_configure_logging_verbose(self, mock_basic_config):
        configure_logging(verbose=True, log_level=None)
        mock_basic_config.assert_called_once_with(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

    @patch("reddit_scrape.cli.logging.basicConfig")
    def test_configure_logging_explicit_level(self, mock_basic_config):
        configure_logging(verbose=False, log_level="DEBUG")
        mock_basic_config.assert_called_once_with(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

    @patch("reddit_scrape.cli.logging.basicConfig")
    def test_configure_logging_noop_when_default(self, mock_basic_config):
        configure_logging(verbose=False, log_level=None)
        mock_basic_config.assert_not_called()

    @patch("reddit_scrape.cli.configure_logging")
    @patch("reddit_scrape.cli.validate_environment")
    @patch("reddit_scrape.cli.PollerService")
    @patch("reddit_scrape.cli.init_db")
    def test_cli_run_passes_explicit_batch_size(
        self, mock_init_db, mock_poller_cls, mock_validate_env, mock_config_logging
    ):
        mock_validate_env.return_value = ServiceConfig(
            reddit_client_id="id",
            reddit_client_secret="sec",
            database_url="postgresql+psycopg://db",
            deepseek_api_key="key",
            llm_batch_size=5,
        )
        mock_service = MagicMock()
        mock_service.run_once.return_value = CycleStats(
            new_stats=MagicMock(inserted=0, updated=0),
            hot_stats=MagicMock(inserted=0, updated=0),
            extracted_count=0,
        )
        mock_poller_cls.return_value = mock_service

        exit_code = main(["run", "--once", "-v", "--llm-batch-size", "8"])
        assert exit_code == 0
        mock_config_logging.assert_called_once_with(verbose=True, log_level=None)
        mock_poller_cls.assert_called_once_with(
            poll_interval_seconds=300,
            hot_poll_interval_seconds=900,
            new_limit=50,
            hot_limit=50,
            thinking_level="low",
            llm_batch_size=8,
        )

