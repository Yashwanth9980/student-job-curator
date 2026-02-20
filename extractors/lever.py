"""
extractors/lever.py
───────────────────
Extractor for Lever-hosted career boards (https://jobs.lever.co/{company}).

DOM structure (validated 2024-Q4)
──────────────────────────────────
Lever renders a React SPA.  After JS hydration, the listing page looks like:

    <div class="postings-group">          ← department group
      <div class="large-category-label"> ← department name (sibling, not parent)
      <div class="posting">             ← individual job card
        <a class="posting-title" href="https://jobs.lever.co/acme/uuid">
          <h5 data-qa="posting-name">Software Engineer Intern</h5>
          <div class="posting-categories">
            <span class="sort-by-location">San Francisco, CA</span>
            <span class="sort-by-team">Engineering</span>
            <span class="sort-by-commitment">Full-time</span>
          </div>
        </a>
        <div class="posting-btn-submit"> ← "Apply" button
      </div>
    </div>

Detail page (https://jobs.lever.co/acme/{uuid}):
    <div class="content">  ← full job description (HTML + text)

Caveats
───────
* The department label lives as a sibling to `.posting`, not as a parent.
  We scrape team from the `span.sort-by-team` inside each card instead.
* Some boards wrap multiple `.postings-group` divs; we flatten them.
* We apply a 500 ms artificial delay between detail-page fetches to avoid
  hammering Lever's CDN.
"""

import asyncio
import logging
import re

from playwright.async_api import Page
from playwright.async_api import TimeoutError as PlaywrightTimeout

from .base import BaseExtractor, RawJob

logger = logging.getLogger(__name__)

_DETAIL_FETCH_DELAY_S: float = 0.5  # courtesy delay between detail requests


class LeverExtractor(BaseExtractor):
    """Scrapes entry-level and internship postings from a Lever job board."""

    PLATFORM_NAME = "lever"
    _BASE_URL = "https://jobs.lever.co"

    # ── CSS selectors ──────────────────────────────────────────────────────
    _SEL_POSTING = "div.posting"
    _SEL_TITLE_TEXT = "h5[data-qa='posting-name']"
    _SEL_TITLE_LINK = "a.posting-title"
    _SEL_LOCATION = "span.sort-by-location"
    _SEL_TEAM = "span.sort-by-team"
    _SEL_COMMITMENT = "span.sort-by-commitment"  # Full-time / Internship / …
    _SEL_DESCRIPTION = "div.content"             # On the detail page

    # ------------------------------------------------------------------

    def _build_board_url(self) -> str:
        return f"{self._BASE_URL}/{self.company_slug}"

    async def _parse_listings(self, page: Page) -> list[RawJob]:
        """Wait for postings to render, then parse every job card."""
        try:
            await page.wait_for_selector(
                self._SEL_POSTING, timeout=self.ACTION_TIMEOUT_MS
            )
        except PlaywrightTimeout:
            self.logger.warning(
                "Selector '%s' not found – board may be empty or JS failed.",
                self._SEL_POSTING,
            )
            return []

        posting_els = await page.query_selector_all(self._SEL_POSTING)
        self.logger.debug("Raw posting elements found: %d", len(posting_els))

        jobs: list[RawJob] = []

        for el in posting_els:
            try:
                # ── Title ──────────────────────────────────────────────
                title_el = await el.query_selector(self._SEL_TITLE_TEXT)
                if not title_el:
                    continue
                title = (await title_el.inner_text()).strip()
                if not title:
                    continue

                # ── URL & job_id ───────────────────────────────────────
                link_el = await el.query_selector(self._SEL_TITLE_LINK)
                href = (await link_el.get_attribute("href") or "") if link_el else ""
                job_url = href.strip()
                job_id = self._extract_job_id(job_url)

                # ── Location ───────────────────────────────────────────
                loc_el = await el.query_selector(self._SEL_LOCATION)
                location = (
                    (await loc_el.inner_text()).strip() if loc_el else "Remote / Not specified"
                )

                # ── Department / Team ──────────────────────────────────
                team_el = await el.query_selector(self._SEL_TEAM)
                department = (
                    (await team_el.inner_text()).strip() if team_el else None
                )

                # ── Employment type ────────────────────────────────────
                commit_el = await el.query_selector(self._SEL_COMMITMENT)
                employment_type = (
                    (await commit_el.inner_text()).strip() if commit_el else None
                )

                jobs.append(
                    RawJob(
                        title=title,
                        company=self.company_name,
                        location=location,
                        url=job_url,
                        platform=self.PLATFORM_NAME,
                        raw_description="",  # populated lazily by fetch_descriptions()
                        job_id=job_id,
                        department=department,
                        employment_type=employment_type,
                    )
                )

            except Exception as exc:
                self.logger.warning("Skipping malformed posting element: %s", exc)

        self.logger.info(
            "Parsed %d listings | company=%s", len(jobs), self.company_slug
        )
        return jobs

    async def fetch_descriptions(
        self, jobs: list[RawJob], page: Page
    ) -> list[RawJob]:
        """
        Visit each job's Lever detail page to populate raw_description.
        Reuses the provided Page object (same browser context as the listing
        scrape) to avoid the overhead of launching a second browser.
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
                    # Graceful fallback: grab all visible body text
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

            # Courtesy delay between requests
            if idx < len(jobs) - 1:
                await asyncio.sleep(_DETAIL_FETCH_DELAY_S)

        return jobs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_job_id(url: str) -> str:
        """
        Pull the UUID from a Lever posting URL.

        Example
        -------
        https://jobs.lever.co/acme/3f2e1a4b-dead-beef-0000-123456789abc
            → '3f2e1a4b-dead-beef-0000-123456789abc'
        """
        match = re.search(
            r"jobs\.lever\.co/[^/]+/([a-zA-Z0-9][a-zA-Z0-9-]+)", url
        )
        return match.group(1) if match else url
