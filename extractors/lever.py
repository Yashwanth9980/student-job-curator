"""
extractors/lever.py
───────────────────
Extractor for Lever-hosted career boards.

Uses the public Lever Postings API instead of a headless browser:
    https://api.lever.co/v0/postings/{company}?mode=json

The API returns a JSON array with all postings in a single request,
including the plain-text description when `mode=json` is used.
No browser required.

API response shape
──────────────────
[
  {
    "id": "3f2e1a4b-dead-beef-0000-123456789abc",
    "text": "Software Engineer Intern",
    "categories": {
      "location": "San Francisco, CA",
      "team":     "Engineering",
      "commitment": "Internship"
    },
    "hostedUrl":        "https://jobs.lever.co/acme/3f2e1a4b-...",
    "descriptionPlain": "We are looking for…",
    "additional":       "…"
  },
  …
]
"""

import logging
import re

import httpx
from playwright.async_api import Page

from .base import BaseExtractor, RawJob

logger = logging.getLogger(__name__)

_API_BASE = "https://api.lever.co/v0/postings"
_REQUEST_TIMEOUT_S: float = 30.0


class LeverExtractor(BaseExtractor):
    """Scrapes entry-level and internship postings from a Lever job board."""

    PLATFORM_NAME = "lever"

    # ------------------------------------------------------------------
    # Abstract method – not used for API-based extraction
    # ------------------------------------------------------------------

    def _build_board_url(self) -> str:
        return f"https://jobs.lever.co/{self.company_slug}"

    async def _parse_listings(self, page: Page) -> list[RawJob]:
        # Not called when scrape() is overridden below.
        return []

    # ------------------------------------------------------------------
    # Override scrape() to use the JSON API instead of Playwright
    # ------------------------------------------------------------------

    async def scrape(self, include_descriptions: bool = False) -> list[RawJob]:
        """
        Fetch all postings from the Lever Postings API.

        Parameters
        ----------
        include_descriptions:
            If True, descriptionPlain is stored in raw_description.
            The API returns it in the same request so there is no
            extra cost.
        """
        api_url = f"{_API_BASE}/{self.company_slug}?mode=json"
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
        for item in data:
            title = (item.get("text") or "").strip()
            if not title:
                continue

            job_id = item.get("id", "")
            job_url = item.get("hostedUrl", "")
            categories = item.get("categories") or {}
            location = categories.get("location") or "Remote / Not specified"
            department = categories.get("team")
            employment_type = categories.get("commitment")

            raw_description = (
                item.get("descriptionPlain", "") if include_descriptions else ""
            )

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
                    employment_type=employment_type,
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
        match = re.search(
            r"jobs\.lever\.co/[^/]+/([a-zA-Z0-9][a-zA-Z0-9-]+)", url
        )
        return match.group(1) if match else url
