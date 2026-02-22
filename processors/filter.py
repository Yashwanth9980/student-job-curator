"""
processors/filter.py
────────────────────
Two-stage filtering pipeline for job listings.

Stage 1 – RegexGate  (CPU-only, zero API cost)
───────────────────────────────────────────────
Inspects the job title and, when available, the description for hard signals.

  REJECT immediately  →  "Senior", "Sr.", "Lead", "Principal", "Staff",
                          "Manager", "Director", or an explicit requirement
                          of ≥ 2 years experience found in the text.

  PASS immediately    →  "Intern", "Internship", "Co-op", "Junior", "Jr.",
                          "Entry-level", "New Grad", "Early Career", etc.

  AMBIGUOUS           →  Neither signal found → forward to Stage 2.

Stage 2 – LLMGate  (Google Gemini API, free tier available)
────────────────────────────────────────────────────────────
Only reached when Stage 1 returns no clear verdict.
Sends (title + truncated description) to Gemini and enforces a strict Pydantic
JSON schema via structured output.  Approves only roles that plausibly
require 0–1 years of professional experience.

Cost controls
─────────────
• MAX_DESCRIPTION_CHARS caps the text sent to the LLM (default 3 000).
• LLM_CONCURRENCY limits simultaneous API calls (default 5).
• Model defaults to gemini-2.0-flash: fast, free-tier eligible.
  Override via env var FILTER_MODEL.
• Conservative fallback on any API error: keep the job rather than silently
  discarding it.

Usage
─────
    from processors.filter import filter_jobs, FilterResult

    results: list[FilterResult] = await filter_jobs(raw_jobs)
    passing  = [r for r in results if r.passed]
    rejected = [r for r in results if not r.passed]
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel

from extractors.base import RawJob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------

# Characters of raw_description forwarded to the LLM (cost control)
MAX_DESCRIPTION_CHARS: int = int(os.getenv("MAX_DESCRIPTION_CHARS", "3000"))

# Max simultaneous Gemini API calls (stays within rate limits)
LLM_CONCURRENCY: int = int(os.getenv("LLM_CONCURRENCY", "5"))

# gemini-2.0-flash: fast, free-tier eligible, sufficient for binary classification.
# Override with FILTER_MODEL=gemini-1.5-pro for higher accuracy.
FILTER_MODEL: str = os.getenv("FILTER_MODEL", "gemini-2.0-flash")

# Gemini API key (required for LLM gate)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")


# ---------------------------------------------------------------------------
# Stage 1 – Regex patterns
# ---------------------------------------------------------------------------

# Title contains any of these words → hard reject
_REJECT_TITLE_RE = re.compile(
    r"\b("
    r"senior|sr\.?|lead|principal|staff|"
    r"manager|director|head\s+of|vp|vice\s+president|"
    r"chief|architect|distinguished|fellow|partner|"
    r"president|officer|exec(?:utive)?"
    r")\b",
    re.IGNORECASE,
)

# Title or description snippet contains these → hard pass
_PASS_KEYWORDS_RE = re.compile(
    r"\b("
    r"intern(?:ship)?|co[.\-]?op|"
    r"entry[.\-]?level|junior|jr\.?|"
    r"new\s*grad(?:uate)?|"
    r"early[.\-]?career|apprentice|trainee|"
    r"associate\s+(?:engineer|developer|analyst|scientist)"
    r")\b",
    re.IGNORECASE,
)

# Explicit years-of-experience patterns that signal a senior role
# Matches: "2-5 years", "3+ years", "at least 3 years", "minimum 2 years"
_YEARS_EXP_RE = re.compile(
    r"(\d+)\s*(?:–|-|to)\s*(\d+)\s*\+?\s*years?"   # range: "2-5 years"
    r"|"
    r"(\d+)\s*\+\s*years?"                           # lower-bound: "3+ years"
    r"|"
    r"(?:at\s+least|minimum\s+of?)\s+(\d+)\s+years?",  # "at least 3 years"
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class Gate(str, Enum):
    REGEX_PASS   = "regex_pass"    # Hard-passed by keyword
    REGEX_REJECT = "regex_reject"  # Hard-rejected by keyword or years
    LLM_PASS     = "llm_pass"      # LLM approved
    LLM_REJECT   = "llm_reject"    # LLM rejected
    LLM_ERROR    = "llm_error"     # API failure – kept conservatively


@dataclass
class FilterResult:
    """Outcome of the two-stage filter for a single job."""
    job: RawJob
    passed: bool
    gate: Gate
    reasoning: str


# ---------------------------------------------------------------------------
# Pydantic schema enforced by the LLM
# ---------------------------------------------------------------------------

class LLMDecision(BaseModel):
    """Structured output schema returned by the LLM gate."""
    is_entry_level: bool
    max_years_required: int          # Best estimate; 0 when not stated
    confidence: Literal["high", "medium", "low"]
    reasoning: str                   # ≤ 2 sentences explaining the verdict


# ---------------------------------------------------------------------------
# Stage 1 – RegexGate
# ---------------------------------------------------------------------------

class RegexGate:
    """
    Fast, zero-cost pre-filter.

    Returns
    -------
    True   – definite entry-level / internship signal
    False  – definite senior / experienced signal
    None   – ambiguous; escalate to the LLM gate
    """

    def evaluate(self, job: RawJob) -> bool | None:
        title = job.title or ""
        # Scan only the first 5 000 chars of the description to stay fast
        description = (job.raw_description or "")[:5_000]

        # ── Hard reject: senior keywords in title ─────────────────────────
        if _REJECT_TITLE_RE.search(title):
            logger.debug("REGEX REJECT (title)       | %r", title)
            return False

        # ── Hard reject: explicit ≥2-year requirement in description ───────
        if description and self._requires_senior_experience(description):
            logger.debug("REGEX REJECT (years in desc) | %r", title)
            return False

        # ── Hard pass: entry-level keywords in title ───────────────────────
        if _PASS_KEYWORDS_RE.search(title):
            logger.debug("REGEX PASS   (title)       | %r", title)
            return True

        # ── Hard pass: entry-level keywords near the top of the description ─
        if description and _PASS_KEYWORDS_RE.search(description[:500]):
            logger.debug("REGEX PASS   (desc header) | %r", title)
            return True

        logger.debug("REGEX AMBIGUOUS            | %r → forwarding to LLM", title)
        return None

    @staticmethod
    def _requires_senior_experience(text: str) -> bool:
        """Return True if text contains an explicit ≥2-year requirement."""
        for match in _YEARS_EXP_RE.finditer(text):
            groups = match.groups()
            # The minimum years value lives in whichever group matched
            min_years = next((int(g) for g in groups if g is not None), 0)
            if min_years >= 2:
                return True
        return False


# ---------------------------------------------------------------------------
# Stage 2 – LLMGate
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a job-board classifier for a student and new-graduate job board.

TASK
────
Decide whether a job listing is appropriate for candidates with 0–1 years
of professional experience: students, recent graduates, or people making
their first career step with no prior relevant work history.

ENTRY-LEVEL (is_entry_level = true) when the role:
  • Explicitly states 0–1 years or "no experience required"
  • Is an internship, co-op, apprenticeship, or graduate programme
  • Has responsibilities appropriate for someone learning on the job
  • Does not require domain expertise that takes years to acquire

NOT ENTRY-LEVEL (is_entry_level = false) when the role:
  • Requires ≥ 2 years of professional experience
  • Lists responsibilities that clearly assume prior industry exposure
  • Uses seniority language or compensation consistent with experienced hires

UNCERTAINTY
───────────
If evidence is absent or contradictory, lean toward is_entry_level = true
and set confidence = "low". Never drop a job due to insufficient data.

Respond ONLY with valid JSON matching the provided schema.\
"""


class LLMGate:
    """
    Semantic classifier backed by Gemini with Pydantic-enforced JSON output.

    A module-level semaphore caps concurrent API calls so we stay within
    Gemini's rate limits even when many ambiguous jobs arrive at once.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

    async def evaluate(self, job: RawJob) -> LLMDecision:
        """
        Call the Gemini API and return a validated LLMDecision.

        On any API failure the method logs a warning and returns a conservative
        pass (is_entry_level=True) so ambiguous jobs are never silently dropped.
        """
        prompt = self._build_prompt(job)

        async with self._semaphore:
            try:
                response = await self._client.aio.models.generate_content(
                    model=FILTER_MODEL,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=LLMDecision,
                        max_output_tokens=256,
                    ),
                )
                decision = LLMDecision.model_validate_json(response.text)
                logger.debug(
                    "LLM decision | title=%r entry=%s years=%d conf=%s",
                    job.title,
                    decision.is_entry_level,
                    decision.max_years_required,
                    decision.confidence,
                )
                return decision

            except genai_errors.ClientError as exc:
                if "429" in str(exc) or "quota" in str(exc).lower():
                    logger.warning(
                        "Rate limit hit | job_id=%s – keeping conservatively",
                        job.job_id,
                    )
                else:
                    logger.warning(
                        "Client error   | job_id=%s err=%s – keeping conservatively",
                        job.job_id, exc,
                    )
            except genai_errors.ServerError as exc:
                logger.warning(
                    "Server error   | job_id=%s err=%s – keeping conservatively",
                    job.job_id, exc,
                )
            except Exception:
                logger.exception(
                    "Unexpected LLM error | job_id=%s – keeping conservatively",
                    job.job_id,
                )

        # Conservative fallback: keep the job, do not silently discard it
        return LLMDecision(
            is_entry_level=True,
            max_years_required=0,
            confidence="low",
            reasoning="LLM unavailable – kept conservatively to avoid data loss.",
        )

    @staticmethod
    def _build_prompt(job: RawJob) -> str:
        """Assemble a compact, token-efficient prompt for a single job."""
        description = (job.raw_description or "").strip()
        if len(description) > MAX_DESCRIPTION_CHARS:
            description = description[:MAX_DESCRIPTION_CHARS] + "\n[description truncated]"

        lines = [
            f"Job Title:  {job.title}",
            f"Company:    {job.company}",
            f"Location:   {job.location}",
        ]
        if job.department:
            lines.append(f"Department: {job.department}")
        if job.employment_type:
            lines.append(f"Work Type:  {job.employment_type}")

        if description:
            lines.append(f"\nJob Description:\n{description}")
        else:
            lines.append("\n(No description provided – classify from title only.)")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singletons  (created once; event loop-safe in Python 3.10+)
# ---------------------------------------------------------------------------

_regex_gate = RegexGate()
_llm_gate   = LLMGate()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def filter_jobs(jobs: list[RawJob]) -> list[FilterResult]:
    """
    Run every RawJob through the two-stage filter pipeline concurrently.

    Returns a FilterResult for **every** input job (passed and rejected) so
    callers can audit what was dropped and why without any silent data loss.

    Parameters
    ----------
    jobs:
        Raw job listings from the extraction layer.

    Returns
    -------
    list[FilterResult]:
        One result per input job.  Check ``result.passed`` to determine
        eligibility.  ``result.gate`` and ``result.reasoning`` explain why.
    """
    if not jobs:
        logger.info("Filter pipeline: no jobs to process.")
        return []

    logger.info("Filter pipeline starting | total_jobs=%d", len(jobs))

    tasks = [_evaluate_one(job) for job in jobs]
    results: list[FilterResult] = await asyncio.gather(*tasks)

    _log_summary(results)
    return results


async def _evaluate_one(job: RawJob) -> FilterResult:
    """Apply Stage 1, then Stage 2 if needed, for a single job."""
    regex_verdict = _regex_gate.evaluate(job)

    if regex_verdict is True:
        return FilterResult(
            job=job,
            passed=True,
            gate=Gate.REGEX_PASS,
            reasoning="Matched an entry-level / internship keyword pattern.",
        )

    if regex_verdict is False:
        return FilterResult(
            job=job,
            passed=False,
            gate=Gate.REGEX_REJECT,
            reasoning=(
                "Matched a senior-role keyword or explicit ≥2-year "
                "experience requirement."
            ),
        )

    # Ambiguous title – pay for LLM analysis
    decision = await _llm_gate.evaluate(job)

    if "LLM unavailable" in decision.reasoning:
        gate = Gate.LLM_ERROR
    elif decision.is_entry_level:
        gate = Gate.LLM_PASS
    else:
        gate = Gate.LLM_REJECT

    return FilterResult(
        job=job,
        passed=decision.is_entry_level,
        gate=gate,
        reasoning=decision.reasoning,
    )


def _log_summary(results: list[FilterResult]) -> None:
    """Emit a single structured INFO line with gate-level breakdown."""
    counts: dict[Gate, int] = {g: 0 for g in Gate}
    for r in results:
        counts[r.gate] += 1

    passed  = sum(1 for r in results if r.passed)
    total   = len(results)
    llm_calls = counts[Gate.LLM_PASS] + counts[Gate.LLM_REJECT] + counts[Gate.LLM_ERROR]

    logger.info(
        "Filter complete | total=%d passed=%d rejected=%d "
        "| regex_pass=%d regex_reject=%d "
        "| llm_calls=%d llm_pass=%d llm_reject=%d llm_error=%d",
        total, passed, total - passed,
        counts[Gate.REGEX_PASS], counts[Gate.REGEX_REJECT],
        llm_calls,
        counts[Gate.LLM_PASS], counts[Gate.LLM_REJECT], counts[Gate.LLM_ERROR],
    )
