"""
database/models.py
──────────────────
SQLAlchemy 2.0 ORM model for the ``jobs`` table.

Schema notes
────────────
• Composite primary key (job_id, platform) ensures uniqueness across multiple
  job boards that may internally use the same numeric or UUID identifiers.

• first_seen_at records when we first encountered the listing and is never
  overwritten on subsequent upserts.  last_seen_at is bumped on every pipeline
  run.  The gap between them lets downstream queries detect stale (closed)
  roles: a job not seen for 30+ days is a safe deletion candidate.

• filter_gate and filter_reason carry Phase B provenance.  Storing them here
  lets operators audit classifier decisions and build retraining datasets by
  querying, e.g., all rows where filter_gate = 'llm_pass'.
"""

from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    # ── Primary key ──────────────────────────────────────────────────────────
    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    platform: Mapped[str] = mapped_column(String, primary_key=True)

    # ── Core listing data ────────────────────────────────────────────────────
    title: Mapped[str] = mapped_column(String, nullable=False)
    company: Mapped[str] = mapped_column(String, nullable=False)
    location: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    department: Mapped[str | None] = mapped_column(String, nullable=True)
    employment_type: Mapped[str | None] = mapped_column(String, nullable=True)
    raw_description: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # ── Phase B provenance ───────────────────────────────────────────────────
    filter_gate: Mapped[str] = mapped_column(String, nullable=False)
    filter_reason: Mapped[str] = mapped_column(Text, nullable=False)

    # ── Timestamps ───────────────────────────────────────────────────────────
    # scraped_at  – when the extractor fetched this listing
    # first_seen_at – set once on the initial INSERT, never overwritten
    # last_seen_at  – updated on every subsequent pipeline run
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Job platform={self.platform!r} job_id={self.job_id!r} "
            f"title={self.title!r} company={self.company!r}>"
        )
