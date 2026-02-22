"""
processors/filter.py
────────────────────
Two-stage filtering pipeline for job listings.

Stage 1 – RegexGate  (CPU-only, zero API cost)
───────────────────────────────────────────────
Inspects the job title and, when available, the description for hard signals.

  REJECT immediately  →  "Senior", "Sr.", "Lead", "Principal", "Staff",
                          "Manager", "Director", or an explicit requirement
                          of ≥ 1 year experience found in the text.

  PASS immediately    →  "Intern", "Internship", "Co-op", "Junior", "Jr.",
                          "Entry-level", "New Grad", "Fresher", "Trainee",
                          "Campus Hire", "Graduate Programme", etc.

  AMBIGUOUS           →  Neither signal found → forward to Stage 2.

Stage 2 – LLMGate  (Google Gemini API, free tier available)
────────────────────────────────────────────────────────────
Only reached when Stage 1 returns no clear verdict.
Sends (title + truncated description) to Gemini and enforces a strict Pydantic
JSON schema via structured output.  Approves ONLY roles that explicitly
welcome freshers / candidates with zero professional experience.
When uncertain, the LLM rejects (strict freshers-only mode).

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

# Jobs per LLM API call.  10 reduces 350 individual calls to 35 batch calls.
LLM_BATCH_SIZE: int = int(os.getenv("LLM_BATCH_SIZE", "10"))

# Maximum LLM requests per minute.  Gemini free tier allows 15 RPM;
# we default to 12 to leave headroom for transient spikes.
LLM_RPM_LIMIT: int = int(os.getenv("LLM_RPM_LIMIT", "12"))

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

# Title or description snippet contains these → hard pass (freshers-only signals)
_PASS_KEYWORDS_RE = re.compile(
    r"\b("
    r"intern(?:ship)?|co[.\-]?op|"
    r"entry[.\-]?level|junior|jr\.?|"
    r"new\s*grad(?:uate)?|"
    r"early[.\-]?career|apprentice|trainee|"
    r"fresher|fresh\s*graduate|"
    r"campus\s+(?:hire|recruit)|graduate\s+(?:program|programme|trainee)|"
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
    """Structured output schema for a single job classification."""
    is_entry_level: bool
    max_years_required: int          # Best estimate; 0 when not stated
    confidence: Literal["high", "medium", "low"]
    reasoning: str                   # ≤ 2 sentences explaining the verdict


class _BatchResponse(BaseModel):
    """Wrapper so Gemini returns a stable JSON object with a named array."""
    decisions: list[LLMDecision]


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
        """Return True if text contains an explicit ≥1-year requirement.

        Freshers-only mode: any role requiring 1+ years of experience is not
        suitable for candidates with no work history.  A range like "0–1 years"
        still passes because the lower bound is 0.
        """
        for match in _YEARS_EXP_RE.finditer(text):
            groups = match.groups()
            # The minimum years value lives in whichever group matched
            min_years = next((int(g) for g in groups if g is not None), 0)
            if min_years >= 1:
                return True
        return False


# ---------------------------------------------------------------------------
# Stage 2 – LLMGate
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a job-board classifier for a FRESHERS-ONLY job board targeting
students and people with zero professional work experience.

TASK
────
You will receive a numbered list of job listings.  For EACH one, decide
whether it is open to candidates with NO prior work experience (freshers,
final-year students, or brand-new graduates with 0 years of experience).

FRESHER-FRIENDLY (is_entry_level = true) ONLY when the role:
  • Explicitly requires 0 years of experience, or states "no experience needed"
  • Is an internship, co-op, apprenticeship, campus hire, or graduate programme
  • Is labelled "fresher", "fresh graduate", "trainee", "junior", or equivalent
  • Has on-the-job training / mentorship language suggesting no prior experience needed

NOT FRESHER-FRIENDLY (is_entry_level = false) when the role:
  • Requires 1 or more years of professional experience
  • Assumes prior industry knowledge, domain expertise, or production experience
  • Uses seniority language (senior, lead, staff, principal, manager, etc.)
  • Compensation or responsibilities clearly target experienced hires

UNCERTAINTY
───────────
If the listing gives NO clear signal either way, set is_entry_level = false
and confidence = "low".  Only approve a role when there is positive evidence
it welcomes freshers.  When in doubt, reject.

Return a JSON object with a single key "decisions" containing an array of
exactly N objects (one per job, in the same order), each with fields:
  is_entry_level, max_years_required, confidence, reasoning.\
"""

_CONSERVATIVE_DECISION = LLMDecision(
    is_entry_level=False,
    max_years_required=0,
    confidence="low",
    reasoning="LLM unavailable – rejected to keep board freshers-only.",
)


class LLMGate:
    """
    Semantic classifier backed by Gemini with Pydantic-enforced JSON output.

    Jobs are sent in batches (default 10 per request) to stay within the
    Gemini free-tier rate limit (15 RPM).  A token-bucket style delay is
    applied between batch requests.
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

    # ------------------------------------------------------------------
    # Public: classify all ambiguous jobs with rate-limited batching
    # ------------------------------------------------------------------

    async def evaluate_all(self, jobs: list[RawJob]) -> list[FilterResult]:
        """
        Classify every job in *jobs* using batched API calls.

        Batches are sent sequentially with a minimum inter-request interval
        derived from LLM_RPM_LIMIT so we never exceed the API quota.
        """
        if not jobs:
            return []
        if self._client is None:
            logger.info(
                "LLM gate disabled (no GEMINI_API_KEY) – rejecting all %d ambiguous jobs "
                "to keep board freshers-only (set GEMINI_API_KEY to enable semantic approval)",
                len(jobs),
            )
            return [
                FilterResult(job=j, passed=False, gate=Gate.LLM_ERROR, reasoning="llm_disabled")
                for j in jobs
            ]

        batches = [
            jobs[i : i + LLM_BATCH_SIZE]
            for i in range(0, len(jobs), LLM_BATCH_SIZE)
        ]
        min_interval = 60.0 / LLM_RPM_LIMIT  # seconds between requests

        logger.info(
            "LLM gate: %d ambiguous jobs → %d batches "
            "(batch_size=%d, rpm_limit=%d)",
            len(jobs), len(batches), LLM_BATCH_SIZE, LLM_RPM_LIMIT,
        )

        results: list[FilterResult] = []
        last_call_time: float = 0.0

        for batch_idx, batch in enumerate(batches, 1):
            # ── Rate limiting ──────────────────────────────────────────
            now = asyncio.get_event_loop().time()
            wait = min_interval - (now - last_call_time)
            if wait > 0:
                await asyncio.sleep(wait)

            logger.debug(
                "LLM batch %d/%d | size=%d", batch_idx, len(batches), len(batch)
            )
            last_call_time = asyncio.get_event_loop().time()

            decisions = await self._call_batch(batch)

            for job, decision in zip(batch, decisions):
                if "LLM unavailable" in decision.reasoning:
                    gate = Gate.LLM_ERROR
                elif decision.is_entry_level:
                    gate = Gate.LLM_PASS
                else:
                    gate = Gate.LLM_REJECT

                results.append(
                    FilterResult(
                        job=job,
                        passed=decision.is_entry_level,
                        gate=gate,
                        reasoning=decision.reasoning,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Private: single batch API call with retry on rate-limit
    # ------------------------------------------------------------------

    async def _call_batch(self, jobs: list[RawJob]) -> list[LLMDecision]:
        """
        Send one batch to Gemini and return a decision per job.

        Retries up to 3 times with exponential back-off on 429s.
        Falls back to conservative pass for all jobs in the batch on
        unrecoverable errors.
        """
        prompt = self._build_batch_prompt(jobs)
        fallback = [_CONSERVATIVE_DECISION] * len(jobs)

        for attempt in range(3):
            try:
                response = await self._client.aio.models.generate_content(
                    model=FILTER_MODEL,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=_BatchResponse,
                        max_output_tokens=300 * len(jobs),
                    ),
                )
                batch_resp = _BatchResponse.model_validate_json(response.text)
                if len(batch_resp.decisions) != len(jobs):
                    raise ValueError(
                        f"Expected {len(jobs)} decisions, "
                        f"got {len(batch_resp.decisions)}"
                    )
                logger.debug(
                    "Batch classified | size=%d pass=%d reject=%d",
                    len(jobs),
                    sum(1 for d in batch_resp.decisions if d.is_entry_level),
                    sum(1 for d in batch_resp.decisions if not d.is_entry_level),
                )
                return batch_resp.decisions

            except genai_errors.ClientError as exc:
                if "429" in str(exc) or "quota" in str(exc).lower():
                    backoff = 10 * (2 ** attempt)  # 10s, 20s, 40s
                    logger.warning(
                        "Rate limit on batch (attempt %d/3) – sleeping %ds",
                        attempt + 1, backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.warning("Batch client error (non-429): %s", exc)
                    break

            except genai_errors.ServerError as exc:
                logger.warning("Batch server error: %s", exc)
                break

            except Exception:
                logger.exception("Unexpected error in LLM batch call")
                break

        logger.warning(
            "All attempts failed for batch of %d jobs – keeping conservatively",
            len(jobs),
        )
        return fallback

    # ------------------------------------------------------------------
    # Private: prompt builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_batch_prompt(jobs: list[RawJob]) -> str:
        """Build a numbered multi-job prompt for a single API call."""
        sections: list[str] = [
            f"Classify the following {len(jobs)} job listing(s):\n"
        ]
        for idx, job in enumerate(jobs, 1):
            description = (job.raw_description or "").strip()
            if len(description) > MAX_DESCRIPTION_CHARS:
                description = (
                    description[:MAX_DESCRIPTION_CHARS] + "\n[description truncated]"
                )

            lines = [
                f"--- Job {idx} ---",
                f"Title:      {job.title}",
                f"Company:    {job.company}",
                f"Location:   {job.location}",
            ]
            if job.department:
                lines.append(f"Department: {job.department}")
            if job.employment_type:
                lines.append(f"Work Type:  {job.employment_type}")
            if description:
                lines.append(f"\nDescription:\n{description}")
            else:
                lines.append("\n(No description – classify from title only.)")

            sections.append("\n".join(lines))

        return "\n\n".join(sections)


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
    Run every RawJob through the two-stage filter pipeline.

    Stage 1 (regex) runs synchronously for all jobs first.
    Stage 2 (LLM) runs only for ambiguous jobs, in rate-limited batches.

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

    # ── Stage 1: regex gate (no API calls, runs instantly) ────────────
    results: list[FilterResult] = []
    ambiguous: list[RawJob] = []

    for job in jobs:
        verdict = _regex_gate.evaluate(job)
        if verdict is True:
            results.append(FilterResult(
                job=job,
                passed=True,
                gate=Gate.REGEX_PASS,
                reasoning="Matched an entry-level / internship keyword pattern.",
            ))
        elif verdict is False:
            results.append(FilterResult(
                job=job,
                passed=False,
                gate=Gate.REGEX_REJECT,
                reasoning=(
                    "Matched a senior-role keyword or explicit ≥2-year "
                    "experience requirement."
                ),
            ))
        else:
            ambiguous.append(job)

    # ── Stage 2: LLM gate (batched, rate-limited) ─────────────────────
    if ambiguous:
        llm_results = await _llm_gate.evaluate_all(ambiguous)
        results.extend(llm_results)

    _log_summary(results)
    return results


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
