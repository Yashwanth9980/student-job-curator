"""
extractors/base.py
──────────────────
Abstract base class shared by every job-board extractor, plus the RawJob
dataclass that travels through the full ETL pipeline.

Design notes
------------
* Playwright is launched in headless Chromium with playwright-stealth applied
  to mask the most common bot-detection signals.
* Heavy assets (images, fonts, media) are blocked to speed up page loads.
* Navigation uses a two-stage wait: domcontentloaded → networkidle.  If the
  idle wait times out the extractor attempts a best-effort parse rather than
  aborting, because partially-loaded job boards still yield useful data.
* `scrape()` is intentionally non-raising: all exceptions are logged and an
  empty list is returned.  This lets the orchestrator run every target even
  when individual boards fail.
* Description fetching is opt-in via `include_descriptions=True` so the
  orchestrator can skip it when testing the listing stage alone.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
)
from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright_stealth import stealth_async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RawJob:
    """
    Normalised, platform-agnostic representation of a single job listing.

    `raw_description` starts empty and is populated by fetch_descriptions()
    before the record reaches the filtering layer (Phase B).
    """

    title: str
    company: str
    location: str
    url: str
    platform: str           # 'lever' | 'greenhouse' | …
    raw_description: str
    job_id: str             # Platform-specific identifier (UUID or numeric)
    department: Optional[str] = None
    employment_type: Optional[str] = None
    scraped_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Base extractor
# ---------------------------------------------------------------------------

class BaseExtractor(ABC):
    """
    Abstract base class for all job-board extractors.

    Concrete subclasses must implement:
        _build_board_url()  → str
        _parse_listings()   → list[RawJob]

    Subclasses may override:
        fetch_descriptions() to populate raw_description from detail pages.
    """

    PLATFORM_NAME: str = ""

    # Playwright timing constants (milliseconds)
    NAV_TIMEOUT_MS: int = 60_000   # max wait for page.goto()
    IDLE_TIMEOUT_MS: int = 15_000  # max wait for networkidle after navigation
    ACTION_TIMEOUT_MS: int = 15_000  # max wait for individual selectors

    # Glob patterns for assets we can safely block to speed up loads
    _BLOCKED_RESOURCES = (
        "**/*.{png,jpg,jpeg,gif,svg,webp,ico,woff,woff2,ttf,eot,mp4,mp3,wav}"
    )

    def __init__(self, company_slug: str, company_name: str) -> None:
        self.company_slug = company_slug
        self.company_name = company_name
        self.logger = logging.getLogger(
            f"{__name__}.{self.PLATFORM_NAME}.{company_slug}"
        )

    # ------------------------------------------------------------------
    # Abstract interface – subclasses must implement both methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_board_url(self) -> str:
        """Return the canonical job-board URL for this company."""

    @abstractmethod
    async def _parse_listings(self, page: Page) -> list[RawJob]:
        """
        Parse the fully-loaded board page and return a list of RawJob objects.
        Called while the browser session is still open.
        """

    # ------------------------------------------------------------------
    # Optional override – description fetching
    # ------------------------------------------------------------------

    async def fetch_descriptions(
        self, jobs: list[RawJob], page: Page
    ) -> list[RawJob]:
        """
        Populate raw_description for each job by visiting its detail URL.
        Default is a no-op; override in subclasses that need it.
        """
        return jobs

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def scrape(self, include_descriptions: bool = False) -> list[RawJob]:
        """
        Launch a stealth Chromium session, navigate to the board, and return
        extracted jobs.  Always returns a list (empty on total failure).

        Parameters
        ----------
        include_descriptions:
            If True, visit each job's detail page inside the same browser
            session to populate raw_description before returning.
        """
        url = self._build_board_url()
        self.logger.info(
            "Scrape started | platform=%s company=%s url=%s",
            self.PLATFORM_NAME, self.company_slug, url,
        )

        async with async_playwright() as playwright:
            browser: Browser = await playwright.chromium.launch(headless=True)
            try:
                context: BrowserContext = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 800},
                    locale="en-US",
                )
                page: Page = await context.new_page()

                # Apply stealth patches to mask headless Chromium signals
                await stealth_async(page)

                # Block heavy media assets – cuts load time substantially
                await page.route(
                    self._BLOCKED_RESOURCES,
                    lambda route: route.abort(),
                )

                # ── Navigate ──────────────────────────────────────────
                try:
                    await page.goto(
                        url,
                        timeout=self.NAV_TIMEOUT_MS,
                        wait_until="domcontentloaded",
                    )
                    # Wait for JS-rendered content to settle
                    await page.wait_for_load_state(
                        "networkidle", timeout=self.IDLE_TIMEOUT_MS
                    )
                except PlaywrightTimeout:
                    self.logger.warning(
                        "Timeout during navigation/idle for %s; "
                        "attempting best-effort parse on partial load",
                        url,
                    )

                # ── Parse listings ────────────────────────────────────
                jobs = await self._parse_listings(page)

                # ── Optionally fetch detail pages ─────────────────────
                if include_descriptions and jobs:
                    self.logger.info(
                        "Fetching descriptions | company=%s count=%d",
                        self.company_slug, len(jobs),
                    )
                    jobs = await self.fetch_descriptions(jobs, page)

                self.logger.info(
                    "Scrape complete | platform=%s company=%s jobs_found=%d",
                    self.PLATFORM_NAME, self.company_slug, len(jobs),
                )
                return jobs

            except Exception:
                self.logger.exception(
                    "Unhandled error | platform=%s company=%s",
                    self.PLATFORM_NAME, self.company_slug,
                )
                return []

            finally:
                await browser.close()
