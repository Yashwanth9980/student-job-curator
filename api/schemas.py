"""
api/schemas.py
──────────────
Pydantic response models for the FastAPI layer.

Kept separate from the SQLAlchemy ORM models in database/models.py so the
API contract can evolve independently of the storage schema.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, HttpUrl


class JobOut(BaseModel):
    """JSON representation of a single passing job listing."""

    model_config = ConfigDict(from_attributes=True)

    job_id: str
    platform: str
    title: str
    company: str
    location: str
    url: str
    department: str | None
    employment_type: str | None

    # Phase B provenance
    filter_gate: str
    filter_reason: str

    # Timestamps
    scraped_at: datetime
    first_seen_at: datetime
    last_seen_at: datetime


class JobListResponse(BaseModel):
    """Paginated list of jobs with total count for frontend pagination."""

    total: int
    limit: int
    offset: int
    jobs: list[JobOut]


class HealthResponse(BaseModel):
    status: str
    database: str
