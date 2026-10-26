import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
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
