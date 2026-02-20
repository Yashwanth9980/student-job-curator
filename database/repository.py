"""
database/repository.py
──────────────────────
Async repository layer: converts FilterResult objects into database rows using
dialect-aware upsert (INSERT … ON CONFLICT DO UPDATE).

Upsert semantics
────────────────
• first_seen_at  – set once on the initial INSERT; never overwritten.
• last_seen_at   – bumped on every pipeline run; lets downstream queries
                   detect stale (closed) roles (e.g. not seen in 30 days).
• All other columns are overwritten on conflict so that title / URL /
  description changes are picked up on re-scrapes.

The full batch of values is sent as a single INSERT statement so the write
is a single round-trip regardless of how many jobs passed filtering.

Cross-database portability
──────────────────────────
The insert helper is selected at import time based on DATABASE_URL:
  • sqlite+aiosqlite  → sqlalchemy.dialects.sqlite.insert
  • postgresql+asyncpg → sqlalchemy.dialects.postgresql.insert
Both expose an identical on_conflict_do_update() API.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from processors.filter import FilterResult

from .engine import DATABASE_URL, get_session_factory
from .models import Job

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dialect selection – chosen once at import time based on DATABASE_URL
# ---------------------------------------------------------------------------

def _pick_insert():
    url = DATABASE_URL.lower()
    if "postgresql" in url or "asyncpg" in url:
        from sqlalchemy.dialects.postgresql import insert
    else:
        from sqlalchemy.dialects.sqlite import insert
    return insert

_insert = _pick_insert()

# Columns refreshed on every upsert (everything except the PK and first_seen_at)
_UPDATE_COLS = (
    "title",
    "company",
    "location",
    "url",
    "department",
    "employment_type",
    "raw_description",
    "filter_gate",
    "filter_reason",
    "scraped_at",
    "last_seen_at",
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class UpsertSummary:
    """Counts returned by upsert_jobs() for logging and observability."""
    total_received: int    # FilterResults passed in (passing + rejected)
    total_passed: int      # Rows with result.passed == True
    upserted: int          # Rows actually written to the database


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def upsert_jobs(results: list[FilterResult]) -> UpsertSummary:
    """
    Persist all passing FilterResults to the ``jobs`` table.

    Callers may pass the full list from filter_jobs() (passing + rejected) or
    pre-filter it – either way only rows with ``result.passed == True`` are
    written to the database.

    The operation is idempotent: re-running the pipeline over the same board
    updates last_seen_at (and any changed fields) without creating duplicates.

    Parameters
    ----------
    results:
        List of FilterResult objects produced by processors.filter.filter_jobs().

    Returns
    -------
    UpsertSummary
        Counts for logging / observability.
    """
    passing = [r for r in results if r.passed]

    summary = UpsertSummary(
        total_received=len(results),
        total_passed=len(passing),
        upserted=0,
    )

    if not passing:
        logger.info(
            "Repository: nothing to store | 0 of %d filtered jobs passed",
            len(results),
        )
        return summary

    now = datetime.now(timezone.utc)
    rows = [_to_row(result, now) for result in passing]

    stmt = _insert(Job).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["job_id", "platform"],
        # Use stmt.excluded to reference the would-be-inserted values.
        # first_seen_at is intentionally absent – it stays from the original INSERT.
        set_={col: stmt.excluded[col] for col in _UPDATE_COLS},
    )

    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(stmt)
        await session.commit()

    summary.upserted = len(passing)
    logger.info(
        "Repository: upserted %d jobs | %d passed / %d total received",
        summary.upserted, summary.total_passed, summary.total_received,
    )
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_row(result: FilterResult, now: datetime) -> dict:
    """Map a FilterResult to a flat dict matching the Job table columns."""
    job = result.job
    return {
        "job_id":          job.job_id,
        "platform":        job.platform,
        "title":           job.title,
        "company":         job.company,
        "location":        job.location,
        "url":             job.url,
        "department":      job.department,
        "employment_type": job.employment_type,
        "raw_description": job.raw_description or "",
        "filter_gate":     result.gate.value,
        "filter_reason":   result.reasoning,
        "scraped_at":      job.scraped_at,
        # first_seen_at is set here for new rows; preserved on conflict
        # because first_seen_at is NOT in _UPDATE_COLS.
        "first_seen_at":   now,
        "last_seen_at":    now,
    }
