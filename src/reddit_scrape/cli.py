"""Command-line interface for the Reddit scrape pipeline."""

import argparse
import logging
import sys
from collections.abc import Sequence

from reddit_scrape.db import init_db
from reddit_scrape.retention import get_default_retention_days, purge_snapshots
from reddit_scrape.service import PollerService, validate_environment


def configure_logging(verbose: bool = False, log_level: str | None = None) -> None:
    """Configure standard stream logging formatting and level."""
    if verbose or log_level:
        level = getattr(logging, log_level.upper()) if log_level else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )


def create_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="reddit-scrape",
        description="Reddit Vinyl Releases Scraper CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    purge_parser = subparsers.add_parser(
        "purge-snapshots",
        help="Purge historical submission snapshots older than retention threshold",
    )
    purge_parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Retention threshold in days (default: SNAPSHOT_RETENTION_DAYS or 30)",
    )
    purge_parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for chunked deletion (default: 5000)",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run the continuous ingestion poller and extraction service",
    )
    run_parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Run a single poll and extraction cycle and exit",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose stream logging",
    )
    run_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Set explicit logging level",
    )
    run_parser.add_argument(
        "--new-interval",
        type=int,
        default=None,
        help="Polling interval in seconds for /new (default: POLL_INTERVAL_SECONDS or 300)",
    )
    run_parser.add_argument(
        "--hot-interval",
        type=int,
        default=None,
        help="Polling interval in seconds for /hot (default: HOT_POLL_INTERVAL_SECONDS or 900)",
    )
    run_parser.add_argument(
        "--new-limit",
        type=int,
        default=50,
        help="Limit of submissions to fetch per /new poll (default: 50)",
    )
    run_parser.add_argument(
        "--hot-limit",
        type=int,
        default=50,
        help="Limit of submissions to fetch per /hot poll (default: 50)",
    )
    run_parser.add_argument(
        "--thinking-level",
        choices=["none", "minimal", "low", "medium", "high", "max"],
        default=None,
        help="Reasoning effort level for LLM extraction (default: LLM_THINKING_LEVEL or low)",
    )
    run_parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=None,
        help="Submissions batched per LLM extraction call (default: LLM_BATCH_SIZE or 5)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "purge-snapshots":
        days = args.days if args.days is not None else get_default_retention_days()
        deleted = purge_snapshots(retention_days=days, batch_size=args.batch_size)
        print(f"Successfully purged {deleted} snapshots older than {days} days.")
        return 0

    if args.command == "run":
        configure_logging(verbose=args.verbose, log_level=args.log_level)

        try:
            config = validate_environment()
        except ValueError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 1

        init_db()

        new_interval = (
            args.new_interval
            if args.new_interval is not None
            else config.poll_interval_seconds
        )
        hot_interval = (
            args.hot_interval
            if args.hot_interval is not None
            else config.hot_poll_interval_seconds
        )
        thinking_level = args.thinking_level or config.llm_thinking_level
        llm_batch_size = (
            args.llm_batch_size
            if args.llm_batch_size is not None
            else config.llm_batch_size
        )

        service = PollerService(
            poll_interval_seconds=new_interval,
            hot_poll_interval_seconds=hot_interval,
            new_limit=args.new_limit,
            hot_limit=args.hot_limit,
            thinking_level=thinking_level,
            llm_batch_size=llm_batch_size,
        )

        if args.once:
            stats = service.run_once()
            print(
                f"Cycle completed: new_inserted={stats.new_stats.inserted}, "
                f"new_updated={stats.new_stats.updated}, "
                f"hot_inserted={stats.hot_stats.inserted}, "
                f"hot_updated={stats.hot_stats.updated}, "
                f"extracted={stats.extracted_count}"
            )
            return 0

        service.run()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

