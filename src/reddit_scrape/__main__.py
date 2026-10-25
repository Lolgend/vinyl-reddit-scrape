"""Entrypoint for python -m reddit_scrape."""

import sys

from reddit_scrape.cli import main

if __name__ == "__main__":
    sys.exit(main())
