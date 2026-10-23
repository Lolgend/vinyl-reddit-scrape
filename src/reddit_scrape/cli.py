"""Command-line interface for the Reddit scrape pipeline."""

import argparse
import sys
from typing import Optional, Sequence

from reddit_scrape.retention import get_default_retention_days, purge_snapshots


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

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "purge-snapshots":
        days = args.days if args.days is not None else get_default_retention_days()
        deleted = purge_snapshots(retention_days=days, batch_size=args.batch_size)
        print(f"Successfully purged {deleted} snapshots older than {days} days.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
