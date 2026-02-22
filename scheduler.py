"""
scheduler.py – Phase D: Autonomous Pipeline Scheduler
──────────────────────────────────────────────────────
Wraps the full ETL pipeline (A → B → C) in an APScheduler AsyncIOScheduler
and adds Phase E (staleness cleanup) and Phase G (Discord notification)
after every successful run.

Run order per cycle
───────────────────
    1. run_pipeline()           – extract → filter → store   (Phases A+B+C)
    2. delete_stale_jobs()      – remove listings not seen in N days (Phase E)
    3. send_new_jobs_digest()   – post new arrivals to Discord (Phase G)

Usage
─────
    python scheduler.py          # start scheduler; runs indefinitely
    python scheduler.py --now    # same, but run once immediately at startup

Configuration (env vars)
────────────────────────
    SCRAPE_INTERVAL_HOURS   How often to run the pipeline       (default: 6)
    SCRAPE_ON_STARTUP       Run once immediately on start       (default: true)
    STALE_DAYS              Days before a job is considered old (default: 30)
    DISCORD_WEBHOOK_URL     Discord channel webhook; notifications skipped if
                            unset.
"""

import argparse
import asyncio
import logging
import os
import signal
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv

from database.cleanup import delete_stale_jobs
from notifier.discord import send_new_jobs_digest

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-40s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRAPE_INTERVAL_HOURS: int = int(os.getenv("SCRAPE_INTERVAL_HOURS", "6"))
SCRAPE_ON_STARTUP: bool = os.getenv("SCRAPE_ON_STARTUP", "true").lower() == "true"
STALE_DAYS: int = int(os.getenv("STALE_DAYS", "30"))


# ---------------------------------------------------------------------------
# Pipeline cycle
# ---------------------------------------------------------------------------

async def scheduled_run() -> None:
    """
    One complete ETL cycle.

    Records the run-start timestamp BEFORE calling the pipeline so that the
    Discord notifier can query ``first_seen_at >= run_start`` to find exactly
    the jobs discovered in this run.
    """
    # Import here so the module can be imported without triggering Playwright
    from main import main as run_pipeline

    run_start = datetime.now(timezone.utc)
    logger.info("Scheduled run starting | run_start=%s", run_start.isoformat())

    # ── Phase A + B + C ────────────────────────────────────────────────────
    try:
        await run_pipeline()
    except Exception:
        logger.exception(
            "Pipeline raised an unhandled exception – "
            "skipping cleanup and notification for this cycle."
        )
        return

    # ── Phase E: remove stale listings ────────────────────────────────────
    try:
        await delete_stale_jobs(days=STALE_DAYS)
    except Exception:
        logger.exception("Staleness cleanup failed – continuing to notification step.")

    # ── Phase G: notify new jobs ───────────────────────────────────────────
    try:
        await send_new_jobs_digest(since_dt=run_start)
    except Exception:
        logger.exception("Discord notification failed.")

    logger.info("Scheduled run complete.")


# ---------------------------------------------------------------------------
# Scheduler bootstrap
# ---------------------------------------------------------------------------

async def run_scheduler(run_now: bool = False) -> None:
    """Start the APScheduler loop and block until SIGINT / SIGTERM."""
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scheduled_run,
        trigger=IntervalTrigger(hours=SCRAPE_INTERVAL_HOURS),
        id="etl_pipeline",
        name="ETL Pipeline",
        replace_existing=True,
        misfire_grace_time=300,   # tolerate up to 5 min of clock drift / sleep
        coalesce=True,            # skip missed runs rather than stacking them
    )
    scheduler.start()
    logger.info(
        "Scheduler running | interval=%dh stale_days=%d",
        SCRAPE_INTERVAL_HOURS, STALE_DAYS,
    )

    if run_now or SCRAPE_ON_STARTUP:
        logger.info("Running pipeline immediately on startup…")
        await scheduled_run()

    # Wait for SIGINT / SIGTERM
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Job Curator Scheduler")
    parser.add_argument(
        "--now",
        action="store_true",
        help="Run the pipeline immediately at startup (overrides SCRAPE_ON_STARTUP).",
    )
    args = parser.parse_args()

    asyncio.run(run_scheduler(run_now=args.now))
