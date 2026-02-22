"""
extractors/greenhouse.py
────────────────────────
Extractor for Greenhouse-hosted career boards.

Uses the public Greenhouse Boards API instead of a headless browser:
    https://boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true

The API returns a JSON payload with all jobs in a single request, including
full job content when `?content=true` is passed.  No browser required.

API response shape
──────────────────
{
  "jobs": [
    {
      "id": 7654321,
      "title": "Software Engineer Intern",
      "location": {"name": "San Francisco, CA"},
      "departments": [{"name": "Engineering"}],
      "absolute_url": "https://boards.greenhouse.io/acme/jobs/7654321",
      "content": "<p>Full HTML description…</p>",
      "updated_at": "2024-01-15T10:00:00-05:00"
    },
    …
  ]
}
"""

import logging
import re

import httpx
from playwright.async_api import Page

from .base import BaseExtractor, RawJob

logger = logging.getLogger(__name__)

_API_BASE = "https://boards-api.greenhouse.io/v1/boards"
_REQUEST_TIMEOUT_S: float = 30.0


class GreenhouseExtractor(BaseExtractor):
    """Scrapes entry-level and internship postings from a Greenhouse job board."""

    PLATFORM_NAME = "greenhouse"

    # ------------------------------------------------------------------
    # Abstract method – not used for API-based extraction
    # ------------------------------------------------------------------

    def _build_board_url(self) -> str:
        return f"https://boards.greenhouse.io/{self.company_slug}"

    async def _parse_listings(self, page: Page) -> list[RawJob]:
        # Not called when scrape() is overridden below.
        return []

    # ------------------------------------------------------------------
    # Override scrape() to use the JSON API instead of Playwright
    # ------------------------------------------------------------------

    async def scrape(self, include_descriptions: bool = False) -> list[RawJob]:
        """
        Fetch all jobs from the Greenhouse Boards API.

        Parameters
        ----------
        include_descriptions:
            If True, the raw HTML job description is stored in
            raw_description.  The API returns it in the same request so
            there is no extra cost.
        """
        api_url = f"{_API_BASE}/{self.company_slug}/jobs?content=true"
        self.logger.info(
            "Scrape started | platform=%s company=%s url=%s",
            self.PLATFORM_NAME, self.company_slug, api_url,
        )

        try:
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as client:
                resp = await client.get(api_url)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            self.logger.exception(
                "API request failed | platform=%s company=%s",
                self.PLATFORM_NAME, self.company_slug,
            )
            return []

        jobs: list[RawJob] = []
        for item in data.get("jobs", []):
            title = (item.get("title") or "").strip()
            if not title:
                continue

            job_id = str(item.get("id", ""))
            job_url = item.get("absolute_url", "")
            location = (item.get("location") or {}).get(
                "name", "Remote / Not specified"
            )

            departments = item.get("departments") or []
            department = departments[0]["name"] if departments else None

            raw_description = item.get("content", "") if include_descriptions else ""

            jobs.append(
                RawJob(
                    title=title,
                    company=self.company_name,
                    location=location,
                    url=job_url,
                    platform=self.PLATFORM_NAME,
                    raw_description=raw_description,
                    job_id=job_id,
                    department=department,
                )
            )

        self.logger.info(
            "Scrape complete | platform=%s company=%s jobs_found=%d",
            self.PLATFORM_NAME, self.company_slug, len(jobs),
        )
        return jobs

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_job_id(url: str) -> str:
        match = re.search(r"/jobs/(\d+)", url)
        return match.group(1) if match else url
