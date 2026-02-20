"""
database/engine.py
──────────────────
Async SQLAlchemy engine and session factory.

The engine is a module-level singleton; call get_engine() to access it and
call init_db() once at application startup to create all tables.

DATABASE_URL defaults to SQLite (./jobs.db) for local development.
Switch to Postgres for production by setting the env var to:
    postgresql+asyncpg://user:password@host:5432/dbname
"""

import logging
import os
import re

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./jobs.db",
)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return (and lazily create) the module-level async engine."""
    global _engine
    if _engine is None:
        logger.info(
            "Creating database engine | url=%s", _mask_password(DATABASE_URL)
        )
        _engine = create_async_engine(
            DATABASE_URL,
            echo=False,           # set True to log all SQL statements for debugging
            pool_pre_ping=True,   # validate connections before checkout (Postgres)
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return (and lazily create) the async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            expire_on_commit=False,  # keep ORM objects usable after session.commit()
        )
    return _session_factory


async def init_db() -> None:
    """
    Create all database tables if they do not already exist.

    Safe to call on every application startup – it is a no-op when the
    schema is already in place.  Call this before the first upsert_jobs().
    """
    from .models import Base  # local import avoids circular deps at module load

    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info(
        "Database initialised (CREATE TABLE IF NOT EXISTS) | url=%s",
        _mask_password(DATABASE_URL),
    )


def _mask_password(url: str) -> str:
    """Return the URL with any embedded password replaced by ***."""
    return re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", url)
