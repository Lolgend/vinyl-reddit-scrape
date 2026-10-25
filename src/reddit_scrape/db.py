import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


def get_database_url(url: Optional[str] = None) -> str:
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


def get_engine(database_url: Optional[str] = None, **kwargs) -> Engine:
    """Create a SQLAlchemy engine configured with psycopg."""
    url = get_database_url(database_url)
    return create_engine(url, **kwargs)


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """Create a configured sessionmaker bound to engine."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine)


def init_db(engine: Optional[Engine] = None) -> None:
    """Create all tables and indexes defined in Base metadata."""
    if engine is None:
        engine = get_engine()
    # Import models to ensure they are registered on Base.metadata
    from reddit_scrape import models  # noqa: F401

    Base.metadata.create_all(engine)
