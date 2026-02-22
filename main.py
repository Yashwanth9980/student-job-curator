"""
main.py – Niche Job Curator Orchestrator
─────────────────────────────────────────
Drives the full ETL pipeline using asyncio with a Semaphore to cap concurrent
Playwright sessions and avoid IP bans.

Current status
--------------
    Phase A  ✓  Extraction  (extractors/)
    Phase B  ✓  Filtering   (processors/)
    Phase C  ✓  Storage     (database/)

Usage
-----
    pip install -r requirements.txt
    playwright install chromium
    cp .env.example .env   # fill in ANTHROPIC_API_KEY and DATABASE_URL
    python main.py
"""

import asyncio
import logging
import os
import re

from dotenv import load_dotenv

load_dotenv()  # must run before any project imports that read os.getenv() at module level

from database import UpsertSummary, init_db, upsert_jobs
from extractors import GreenhouseExtractor, LeverExtractor, RawJob
from processors import FilterResult, filter_jobs

# ---------------------------------------------------------------------------
# Logging – structured timestamps; no bare print() statements anywhere
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-40s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target company boards
# Each tuple: (company_slug, human_readable_name, platform)
# ---------------------------------------------------------------------------
TARGETS: list[tuple[str, str, str]] = [
    # ── Lever ──────────────────────────────────────────────────────────────
    ("netflix",    "Netflix",    "lever"),
    ("razorpay",   "Razorpay",   "lever"),   # Bangalore HQ
    ("meesho",     "Meesho",     "lever"),   # Bangalore HQ
    # ── Greenhouse ─────────────────────────────────────────────────────────
    ("airbnb",     "Airbnb",     "greenhouse"),
    ("stripe",     "Stripe",     "greenhouse"),
    ("coinbase",   "Coinbase",   "greenhouse"),
    ("freshworks", "Freshworks", "greenhouse"),  # India HQ, large Bangalore office
    ("chargebee",  "Chargebee",  "greenhouse"),  # Bangalore office
    ("uber",       "Uber",       "greenhouse"),  # large Bangalore tech centre
    ("twilio",     "Twilio",     "greenhouse"),  # Bangalore office
]

# ---------------------------------------------------------------------------
# Location filter – only keep jobs whose location mentions Bangalore / India
# ---------------------------------------------------------------------------

# Match any of these tokens (case-insensitive) in job.location
_LOCATION_RE = re.compile(
    r"\b(bangalore|bengaluru|india|karnataka)\b",
    re.IGNORECASE,
)

# Set LOCATION_FILTER=false in .env to disable and see jobs from all regions
LOCATION_FILTER: bool = os.getenv("LOCATION_FILTER", "true").lower() != "false"


def _location_matches(job: RawJob) -> bool:
    """Return True if the job's location is in the Bangalore / India region."""
    return bool(_LOCATION_RE.search(job.location or ""))

# Read from environment; default to 3 concurrent browser sessions
MAX_CONCURRENCY: int = int(os.getenv("MAX_CONCURRENCY", "3"))

# Set to True to visit each job's detail page for the full description.
# Significantly slower (one HTTP round-trip per job), but needed for Phase B.
FETCH_DESCRIPTIONS: bool = os.getenv("FETCH_DESCRIPTIONS", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Per-target runner
# ---------------------------------------------------------------------------

async def run_extractor(
    slug: str,
    name: str,
    platform: str,
    semaphore: asyncio.Semaphore,
) -> list[RawJob]:
    """
    Acquire a concurrency slot, run the appropriate extractor, then release.
    Wraps everything in try/except: a single broken board never kills the
    pipeline.
    """
    async with semaphore:
        logger.info(
            "Slot acquired | company=%s platform=%s", slug, platform
        )
        try:
            if platform == "lever":
                extractor = LeverExtractor(
                    company_slug=slug, company_name=name
                )
            elif platform == "greenhouse":
                extractor = GreenhouseExtractor(
                    company_slug=slug, company_name=name
                )
            else:
                logger.error(
                    "Unknown platform '%s' for company '%s' – skipping",
                    platform, slug,
                )
                return []

            return await extractor.scrape(include_descriptions=FETCH_DESCRIPTIONS)

        except Exception:
            logger.exception(
                "Extractor raised unexpectedly | company=%s platform=%s",
                slug, platform,
            )
            return []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def main() -> None:
    logger.info(
        "Pipeline starting | targets=%d max_concurrency=%d "
        "fetch_descriptions=%s location_filter=%s",
        len(TARGETS), MAX_CONCURRENCY, FETCH_DESCRIPTIONS, LOCATION_FILTER,
    )

    # ── Phase C: initialise database schema (no-op if already exists) ────────
    await init_db()

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Fire all extraction tasks concurrently, bounded by the semaphore
    tasks = [
        run_extractor(slug, name, platform, semaphore)
        for slug, name, platform in TARGETS
    ]
    results: list[list[RawJob]] = await asyncio.gather(*tasks)

    all_jobs: list[RawJob] = [job for batch in results for job in batch]
    logger.info(
        "Extraction complete | total_jobs_extracted=%d", len(all_jobs)
    )

    # ── Location pre-filter (Bangalore / India) ───────────────────────────
    if LOCATION_FILTER:
        before = len(all_jobs)
        all_jobs = [j for j in all_jobs if _location_matches(j)]
        logger.info(
            "Location filter | kept=%d / %d  (region=Bangalore/India)",
            len(all_jobs), before,
        )

    # ── Phase B – filtering ───────────────────────────────────────────────
    filter_results: list[FilterResult] = await filter_jobs(all_jobs)
    passing_jobs = [r.job for r in filter_results if r.passed]
    logger.info(
        "Filtering complete | passed=%d / total=%d",
        len(passing_jobs), len(all_jobs),
    )

    # ── Phase C: persist passing jobs to the database ─────────────────────
    summary: UpsertSummary = await upsert_jobs(filter_results)
    logger.info(
        "Storage complete | upserted=%d passed=%d total_received=%d",
        summary.upserted, summary.total_passed, summary.total_received,
    )

    # ── Preview: passing jobs ─────────────────────────────────────────────
    logger.info("── Passing jobs preview (first 10) ─────────────────────")
    for job in passing_jobs[:10]:
        logger.info(
            "  [%s] %r @ %r  (%s)",
            job.platform.upper(), job.title, job.company, job.location,
        )

    # ── Preview: rejection log ────────────────────────────────────────────
    rejected = [r for r in filter_results if not r.passed]
    if rejected:
        logger.info("── Rejected jobs (first 5) ──────────────────────────")
        for result in rejected[:5]:
            logger.info(
                "  [%-14s] %r  reason=%r",
                result.gate.value, result.job.title, result.reasoning,
            )


if __name__ == "__main__":
    asyncio.run(main())
