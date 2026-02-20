"""
main.py – Niche Job Curator Orchestrator
─────────────────────────────────────────
Drives the full ETL pipeline using asyncio with a Semaphore to cap concurrent
Playwright sessions and avoid IP bans.

Current status
--------------
    Phase A  ✓  Extraction  (this file + extractors/)
    Phase B  ○  Filtering   (processors/ – stub, wired in next iteration)
    Phase C  ○  Storage     (database/   – stub, wired in next iteration)

Usage
-----
    pip install -r requirements.txt
    playwright install chromium
    cp .env.example .env   # then fill in values
    python main.py
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from extractors import GreenhouseExtractor, LeverExtractor, RawJob

load_dotenv()

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
    ("netflix",  "Netflix",  "lever"),
    ("figma",    "Figma",    "lever"),
    ("notion",   "Notion",   "lever"),
    # ── Greenhouse ─────────────────────────────────────────────────────────
    ("airbnb",   "Airbnb",   "greenhouse"),
    ("stripe",   "Stripe",   "greenhouse"),
    ("coinbase", "Coinbase", "greenhouse"),
]

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
        "Pipeline starting | targets=%d max_concurrency=%d fetch_descriptions=%s",
        len(TARGETS), MAX_CONCURRENCY, FETCH_DESCRIPTIONS,
    )

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

    # ── Phase B stub – filtering will be inserted here ────────────────────
    # from processors.filter import filter_jobs
    # filtered_jobs = await filter_jobs(all_jobs)
    filtered_jobs = all_jobs  # pass-through until Phase B is wired in

    # ── Phase C stub – storage will be inserted here ──────────────────────
    # from database.repository import upsert_jobs
    # await upsert_jobs(filtered_jobs)
    logger.info(
        "Storage stub | jobs_to_store=%d (Phase C not yet wired)",
        len(filtered_jobs),
    )

    # ── Preview ───────────────────────────────────────────────────────────
    logger.info("── Job preview (first 10) ──────────────────────────────")
    for job in all_jobs[:10]:
        logger.info(
            "  [%s] %r @ %r  (%s)",
            job.platform.upper(), job.title, job.company, job.location,
        )


if __name__ == "__main__":
    asyncio.run(main())
