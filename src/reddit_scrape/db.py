import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def get_database_url(url: str | None = None) -> str:
    """Retrieve and normalize database URL with psycopg dialect."""
    if not url:
        # Check current and parent directories for .env
        env_path = Path(".env")
        if not env_path.exists():
            env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)

        url = os.environ.get("DATABASE_URL")

    if not url:
        raise ValueError("DATABASE_URL is not set in environment or .env")

    # Ensure SQLAlchemy psycopg dialect format: postgresql+psycopg://
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)

    return url


def get_engine(database_url: str | None = None, **kwargs) -> Engine:
    """Create a SQLAlchemy engine configured with psycopg."""
    url = get_database_url(database_url)
    return create_engine(url, **kwargs)


def get_session_factory(engine: Engine | None = None) -> sessionmaker:
    """Create a configured sessionmaker bound to engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)


def init_db(engine: Engine | None = None, drop_existing: bool = False) -> None:
    """Create all tables and indexes defined in Base metadata.

    If drop_existing is True, drops all existing tables before re-creating them.
    Also detects and drops incompatible legacy 2-tier catalog tables
    (release_artists, legacy flat releases/artists) before create_all.
    """
    if engine is None:
        engine = get_engine()
    # Import models to ensure they are registered on Base.metadata
    from reddit_scrape import models  # noqa: F401

    if drop_existing:
        Base.metadata.drop_all(engine)

    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

            # Check for incompatible legacy 2-tier catalog tables
            inspector = inspect(engine)
            existing_tables = set(inspector.get_table_names())

            is_legacy = False
            if "release_artists" in existing_tables:
                is_legacy = True
            elif "releases" in existing_tables:
                col_names = {c["name"] for c in inspector.get_columns("releases")}
                if "album_id" not in col_names or "embedding" in col_names:
                    is_legacy = True
            elif "artists" in existing_tables:
                col_names = {c["name"] for c in inspector.get_columns("artists")}
                if "discogs_artist_id" not in col_names:
                    is_legacy = True

            if is_legacy:
                # Cleanly drop legacy catalog tables in dependent order
                # Preserves submissions and submission_snapshots
                legacy_drop_sqls = [
                    "DROP TABLE IF EXISTS submission_releases CASCADE;",
                    "DROP TABLE IF EXISTS release_artists CASCADE;",
                    "DROP TABLE IF EXISTS album_artists CASCADE;",
                    "DROP TABLE IF EXISTS releases CASCADE;",
                    "DROP TABLE IF EXISTS albums CASCADE;",
                    "DROP TABLE IF EXISTS artists CASCADE;",
                ]
                for sql in legacy_drop_sqls:
                    conn.execute(text(sql))

    Base.metadata.create_all(engine)


def reset_db(engine: Engine | None = None) -> None:
    """Drop all existing tables, ensure vector extension, and re-create all tables cleanly."""
    init_db(engine=engine, drop_existing=True)


__all__ = [
    "Base",
    "get_database_url",
    "get_engine",
    "get_session_factory",
    "init_db",
    "reset_db",
]
