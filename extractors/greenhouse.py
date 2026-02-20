"""
extractors/greenhouse.py
────────────────────────
Extractor for Greenhouse-hosted career boards (https://boards.greenhouse.io/{company}).

DOM structure (validated 2024-Q4)
──────────────────────────────────
Unlike Lever, Greenhouse renders its listing page as server-side HTML, so we
do not need to wait for heavy JS hydration.  The structure is:

    <section class="level-0">            ← one section per department
      <h3 class="...">Engineering</h3>  ← department heading (h3 or h2)
      <div class="opening">            ← individual job row
        <a href="/acme/jobs/7654321">Job Title</a>
        <span class="location">San Francisco, CA</span>
      </div>
      …
    </section>

Detail page (https://boards.greenhouse.io/{company}/jobs/{id}):
    <div id="content">  ← full job description

Caveats
───────
* Some companies use a custom Greenhouse embed (boards.eu.greenhouse.io or a
  custom subdomain).  `_normalise_url()` handles relative hrefs so we always
  produce absolute URLs.
* If the page has no <section> wrappers (flat HTML variant), `_flat_parse()`
  is used as a fallback.
* We apply a 500 ms courtesy delay between detail-page fetches.
"""

import asyncio
import logging
import re

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeout

from .base import BaseExtractor, RawJob

logger = logging.getLogger(__name__)

_DETAIL_FETCH_DELAY_S: float = 0.5


class GreenhouseExtractor(BaseExtractor):
    """Scrapes entry-level and internship postings from a Greenhouse job board."""

    PLATFORM_NAME = "greenhouse"
    _BASE_URL = "https://boards.greenhouse.io"

    # ── CSS selectors ──────────────────────────────────────────────────────
    _SEL_JOB_ROW = "div.opening"
    _SEL_SECTION = "section"
    _SEL_DEPT_HEADING = "h3, h2"     # department label within a <section>
    _SEL_JOB_LINK = "a"             # first <a> inside .opening
    _SEL_LOCATION = "span.location"
    _SEL_DESCRIPTION = "div#content"  # on the detail page

    # ------------------------------------------------------------------

    def _build_board_url(self) -> str:
        return f"{self._BASE_URL}/{self.company_slug}"

    async def _parse_listings(self, page: Page) -> list[RawJob]:
        """
        Parse the Greenhouse listing page.

        Strategy
        --------
        1. Wait for at least one `.opening` element.
        2. Try section-based parsing (preserves department labels).
        3. Fall back to flat scan if no <section> wrappers exist.
        """
        try:
            await page.wait_for_selector(
                self._SEL_JOB_ROW, timeout=self.ACTION_TIMEOUT_MS
            )
        except PlaywrightTimeout:
            self.logger.warning(
                "Selector '%s' not found – board may be empty or structure changed.",
                self._SEL_JOB_ROW,
            )
            return []

        sections = await page.query_selector_all(self._SEL_SECTION)
        self.logger.debug("Department sections found: %d", len(sections))

        jobs: list[RawJob] = []

        if sections:
            for section in sections:
                dept_el = await section.query_selector(self._SEL_DEPT_HEADING)
                department = (
                    (await dept_el.inner_text()).strip() if dept_el else None
                )

                rows = await section.query_selector_all(self._SEL_JOB_ROW)
                for row in rows:
                    job = await self._parse_opening(row, department)
                    if job:
                        jobs.append(job)
        else:
            self.logger.debug(
                "No <section> wrappers found – using flat .opening scan."
            )
            jobs = await self._flat_parse(page)

        self.logger.info(
            "Parsed %d listings | company=%s", len(jobs), self.company_slug
        )
        return jobs

    async def _parse_opening(
        self, row, department: str | None
    ) -> RawJob | None:
        """Extract a RawJob from a single `.opening` element."""
        try:
            link_el = await row.query_selector(self._SEL_JOB_LINK)
            if not link_el:
                return None

            title = (await link_el.inner_text()).strip()
            if not title:
                return None

            href = (await link_el.get_attribute("href") or "").strip()
            job_url = self._normalise_url(href)
            job_id = self._extract_job_id(job_url)

            loc_el = await row.query_selector(self._SEL_LOCATION)
            location = (
                (await loc_el.inner_text()).strip()
                if loc_el
                else "Remote / Not specified"
            )

            return RawJob(
                title=title,
                company=self.company_name,
                location=location,
                url=job_url,
                platform=self.PLATFORM_NAME,
                raw_description="",  # populated lazily by fetch_descriptions()
                job_id=job_id,
                department=department,
            )

        except Exception as exc:
            self.logger.warning("Skipping malformed .opening row: %s", exc)
            return None

    async def _flat_parse(self, page: Page) -> list[RawJob]:
        """Fallback: iterate all .opening elements without section context."""
        rows = await page.query_selector_all(self._SEL_JOB_ROW)
        jobs: list[RawJob] = []
        for row in rows:
            job = await self._parse_opening(row, department=None)
            if job:
                jobs.append(job)
        return jobs

    async def fetch_descriptions(
        self, jobs: list[RawJob], page: Page
    ) -> list[RawJob]:
        """
        Visit each job's Greenhouse detail page to populate raw_description.
        """
        for idx, job in enumerate(jobs):
            if not job.url:
                continue
            try:
                await page.goto(
                    job.url,
                    timeout=self.NAV_TIMEOUT_MS,
                    wait_until="domcontentloaded",
                )
                await page.wait_for_load_state(
                    "networkidle", timeout=self.IDLE_TIMEOUT_MS
                )

                desc_el = await page.query_selector(self._SEL_DESCRIPTION)
                if desc_el:
                    job.raw_description = (await desc_el.inner_text()).strip()
                else:
                    job.raw_description = (await page.inner_text("body")).strip()

                self.logger.debug(
                    "Description fetched | job_id=%s chars=%d",
                    job.job_id, len(job.raw_description),
                )

            except Exception as exc:
                self.logger.warning(
                    "Could not fetch description | job_id=%s url=%s err=%s",
                    job.job_id, job.url, exc,
                )

            if idx < len(jobs) - 1:
                await asyncio.sleep(_DETAIL_FETCH_DELAY_S)

        return jobs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise_url(self, href: str) -> str:
        """Ensure the job URL is absolute."""
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return f"{self._BASE_URL}{href}"
        # Relative path without leading slash
        return f"{self._build_board_url()}/{href}"

    @staticmethod
    def _extract_job_id(url: str) -> str:
        """
        Pull the numeric job ID from a Greenhouse URL.

        Example
        -------
        https://boards.greenhouse.io/acme/jobs/7654321  →  '7654321'
        """
        match = re.search(r"/jobs/(\d+)", url)
        return match.group(1) if match else url
