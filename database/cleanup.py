"""
database/cleanup.py
───────────────────
Removes job listings that have not been seen in recent pipeline runs.

A job is considered stale when its last_seen_at timestamp is older than
STALE_DAYS days (default 30).  Since last_seen_at is bumped on every
successful upsert, any listing that falls below the cutoff almost certainly
no longer appears on the company's job board (i.e. the role is closed).

Hard-deletion is intentional: closed roles have no value to students browsing
the board, and the first_seen_at / last_seen_at window already encoded the
lifecycle before the row was removed.

Usage
─────
    from database.cleanup import delete_stale_jobs

    deleted: int = await delete_stale_jobs(days=30)
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete

from .engine import get_session_factory
from .models import Job

logger = logging.getLogger(__name__)


async def delete_stale_jobs(days: int = 30) -> int:
    """
    Hard-delete all jobs whose last_seen_at is older than ``days`` days.

    Parameters
    ----------
    days:
        Staleness threshold.  A job not seen in this many days is deleted.
        Defaults to 30; override via the STALE_DAYS env var in the scheduler.

    Returns
    -------
    int
        Number of rows deleted (0 when nothing was stale).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    stmt = delete(Job).where(Job.last_seen_at < cutoff)

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(stmt)
        await session.commit()
        deleted: int = result.rowcount

    if deleted:
        logger.info(
            "Cleanup: deleted %d stale job(s) | not seen since %s",
            deleted,
            cutoff.date().isoformat(),
        )
    else:
        logger.info(
            "Cleanup: no stale jobs found | cutoff=%s", cutoff.date().isoformat()
        )

    return deleted
