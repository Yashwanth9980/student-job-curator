"""
notifier/discord.py
───────────────────
Phase G: Posts a digest of newly-discovered entry-level jobs to a Discord
channel via an incoming webhook.

"New" is defined as first_seen_at >= since_dt (the timestamp recorded at
the start of the pipeline run that produced these jobs).  This makes every
notification exactly accurate: no duplicates across runs, no missed jobs.

Configuration
─────────────
    DISCORD_WEBHOOK_URL     Incoming webhook URL from Discord channel settings.
                            If unset, notifications are silently skipped.
    DISCORD_MAX_EMBEDS      Max embed cards per message (default 10).
                            Discord enforces a hard limit of 10 per payload.

Rate-limit handling
───────────────────
Discord webhooks are limited to ~5 requests / 2 s.  When sending multiple
batches (>10 new jobs) a 1-second pause is inserted between messages.
A 429 response triggers a back-off using the Retry-After header value.

Usage
─────
    from notifier.discord import send_new_jobs_digest
    from datetime import datetime, timezone

    since = datetime.now(timezone.utc)
    await send_new_jobs_digest(since_dt=since)
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from database.engine import get_session_factory
from database.models import Job

logger = logging.getLogger(__name__)

# Discord hard-caps embeds at 10 per message
_MAX_EMBEDS: int = min(int(os.getenv("DISCORD_MAX_EMBEDS", "10")), 10)

# Colours (decimal) for embed left-bar
_COLOUR_BY_GATE: dict[str, int] = {
    "regex_pass": 0x57F287,   # green  – confident entry-level signal
    "llm_pass":   0x5865F2,   # blurple – LLM approved
    "llm_error":  0xFEE75C,   # yellow  – kept conservatively
}
_COLOUR_DEFAULT: int = 0x99AAB5  # grey


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def send_new_jobs_digest(since_dt: datetime) -> int:
    """
    Query the database for jobs first seen since ``since_dt`` and post them
    to Discord as embed cards.

    Parameters
    ----------
    since_dt:
        Inclusive lower bound on first_seen_at.  Typically the timestamp
        recorded at the start of the current pipeline run.

    Returns
    -------
    int
        Number of new jobs included in the digest (0 if none or webhook unset).
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        logger.info("DISCORD_WEBHOOK_URL not set – skipping notification.")
        return 0

    jobs = await _fetch_new_jobs(since_dt)
    if not jobs:
        logger.info(
            "Discord: no new jobs since %s – nothing to send.", since_dt.isoformat()
        )
        return 0

    batches = [jobs[i : i + _MAX_EMBEDS] for i in range(0, len(jobs), _MAX_EMBEDS)]
    total_new = len(jobs)

    async with httpx.AsyncClient(timeout=15) as client:
        for batch_index, batch in enumerate(batches):
            payload = _build_payload(batch, total_new, batch_index, len(batches))
            await _post_with_retry(client, webhook_url, payload)
            if batch_index < len(batches) - 1:
                await asyncio.sleep(1)   # stay within Discord rate limits

    logger.info("Discord: digest sent | new_jobs=%d batches=%d", total_new, len(batches))
    return total_new


# ---------------------------------------------------------------------------
# Database query
# ---------------------------------------------------------------------------

async def _fetch_new_jobs(since_dt: datetime) -> list[Job]:
    """Return all jobs whose first_seen_at >= since_dt, ordered by company+title."""
    stmt = (
        select(Job)
        .where(Job.first_seen_at >= since_dt)
        .order_by(Job.company, Job.title)
    )
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(stmt)
        return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Discord payload builders
# ---------------------------------------------------------------------------

def _build_payload(
    jobs: list[Job],
    total_new: int,
    batch_index: int,
    total_batches: int,
) -> dict:
    """Assemble the Discord webhook JSON payload for one batch of jobs."""
    embeds = [_job_to_embed(job) for job in jobs]

    # Append a summary footer to the last embed in the last batch
    if batch_index == total_batches - 1 and embeds:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        embeds[-1]["footer"] = {
            "text": f"student-job-curator · {total_new} new listing(s) · {ts}"
        }

    return {"embeds": embeds}


def _job_to_embed(job: Job) -> dict:
    """Convert a Job ORM object into a Discord embed dict."""
    colour = _COLOUR_BY_GATE.get(job.filter_gate, _COLOUR_DEFAULT)

    fields = [
        {"name": "Company",  "value": job.company,              "inline": True},
        {"name": "Location", "value": job.location or "—",      "inline": True},
        {"name": "Platform", "value": job.platform.capitalize(), "inline": True},
    ]
    if job.department:
        fields.append({"name": "Department", "value": job.department, "inline": True})
    if job.employment_type:
        fields.append({"name": "Type", "value": job.employment_type, "inline": True})

    return {
        "title": f"{job.title}",
        "url": job.url,
        "color": colour,
        "author": {"name": job.company},
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# HTTP posting with retry on rate-limit (429)
# ---------------------------------------------------------------------------

async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    max_retries: int = 3,
) -> None:
    """POST payload to the Discord webhook; back off on 429, retry up to max_retries."""
    for attempt in range(max_retries + 1):
        response = await client.post(url, json=payload)

        if response.status_code in (200, 204):
            return

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", "2"))
            logger.warning(
                "Discord rate limit hit – waiting %.1fs before retry %d/%d",
                retry_after, attempt + 1, max_retries,
            )
            await asyncio.sleep(retry_after)
            continue

        # Any other non-2xx: log and give up for this batch
        logger.error(
            "Discord webhook returned unexpected status %d | body=%s",
            response.status_code,
            response.text[:200],
        )
        return

    logger.error("Discord: exhausted %d retries – batch not delivered.", max_retries)
