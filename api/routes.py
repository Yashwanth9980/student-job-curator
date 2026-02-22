"""
api/routes.py
─────────────
FastAPI route handlers for the job-listing endpoints.

Endpoints
─────────
    GET  /jobs                  Paginated list of all stored (passing) jobs.
                                Filterable by company, platform, location,
                                filter_gate.  Sorted newest-first by default.
    GET  /jobs/{platform}/{job_id}
                                Single job detail.

Query parameters for GET /jobs
──────────────────────────────
    company     Exact match (case-insensitive)
    platform    "lever" | "greenhouse"
    location    Substring match (case-insensitive)
    gate        "regex_pass" | "llm_pass" | "llm_error"
    limit       Max rows returned      (default 50, max 200)
    offset      Rows to skip           (default 0)
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.engine import get_session_factory
from database.models import Job

from .schemas import JobListResponse, JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Dependency: async database session
# ---------------------------------------------------------------------------

async def get_session() -> AsyncSession:  # type: ignore[return]
    session_factory = get_session_factory()
    async with session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("", response_model=JobListResponse, summary="List entry-level jobs")
async def list_jobs(
    company: str | None = Query(None, description="Exact company name (case-insensitive)"),
    platform: str | None = Query(None, description="Job board platform: lever | greenhouse"),
    location: str | None = Query(None, description="Substring match on location"),
    gate: str | None = Query(None, description="Filter gate: regex_pass | llm_pass | llm_error"),
    limit: int = Query(50, ge=1, le=200, description="Max results per page"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    session: AsyncSession = Depends(get_session),
) -> JobListResponse:
    """Return a paginated, filterable list of entry-level job listings."""
    base_q = select(Job)

    if company:
        base_q = base_q.where(func.lower(Job.company) == company.lower())
    if platform:
        base_q = base_q.where(Job.platform == platform.lower())
    if location:
        base_q = base_q.where(Job.location.ilike(f"%{location}%"))
    if gate:
        base_q = base_q.where(Job.filter_gate == gate)

    # Total count (before pagination)
    count_q = select(func.count()).select_from(base_q.subquery())
    total: int = (await session.execute(count_q)).scalar_one()

    # Paginated results – newest first
    rows_q = (
        base_q
        .order_by(Job.first_seen_at.desc())
        .limit(limit)
        .offset(offset)
    )
    jobs = (await session.execute(rows_q)).scalars().all()

    return JobListResponse(
        total=total,
        limit=limit,
        offset=offset,
        jobs=[JobOut.model_validate(j) for j in jobs],
    )


@router.get(
    "/{platform}/{job_id}",
    response_model=JobOut,
    summary="Get a single job by platform + job_id",
)
async def get_job(
    platform: str,
    job_id: str,
    session: AsyncSession = Depends(get_session),
) -> JobOut:
    """Fetch one job listing by its composite primary key."""
    job = await session.get(Job, (job_id, platform))
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobOut.model_validate(job)
