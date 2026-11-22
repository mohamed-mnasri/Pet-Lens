"""A job queue in Postgres.

Workers claim the oldest runnable job with ``SELECT ... FOR UPDATE SKIP LOCKED``, so two
workers never get the same job. A running job sends heartbeats; a job whose heartbeat
stops (the worker crashed or the PC slept) is put back in the queue after a timeout.
Failed jobs are retried with a growing delay until ``max_attempts``, then marked dead.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from petlens.db import models as m

BASE_RETRY_DELAY = timedelta(seconds=30)
STUCK_AFTER = timedelta(minutes=10)


def _now(session: Session) -> datetime:
    """The database clock, so retries and timeouts don't depend on the PC's clock."""
    now: datetime = session.execute(select(func.now())).scalar_one()
    return now


def enqueue(
    session: Session,
    kind: str,
    payload: dict[str, Any] | None = None,
    max_attempts: int = 3,
    total: int | None = None,
) -> m.Job:
    job = m.Job(kind=kind, payload=payload or {}, max_attempts=max_attempts, progress_total=total)
    session.add(job)
    session.flush()
    session.refresh(job)
    return job


_CLAIM = text(
    """
    UPDATE job SET state = 'running', attempts = attempts + 1, error = NULL,
           started_at = now(), heartbeat_at = now(), updated_at = now()
    WHERE id = (
        SELECT id FROM job
        WHERE state = 'queued' AND run_after <= now() AND (:any_kind OR kind = ANY(:kinds))
        ORDER BY run_after, created_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    )
    RETURNING id
    """
)


def claim(session: Session, kinds: list[str] | None = None) -> m.Job | None:
    """Take the next runnable job, or None. Commits, so the claim is visible at once."""
    job_id = session.execute(
        _CLAIM, {"any_kind": kinds is None, "kinds": kinds or []}
    ).scalar_one_or_none()
    session.commit()
    if job_id is None:
        return None
    return session.get(m.Job, job_id, populate_existing=True)


def heartbeat(
    session: Session,
    job_id: uuid.UUID,
    done: int | None = None,
    total: int | None = None,
    partial: dict[str, Any] | None = None,
) -> None:
    """Mark the job alive, optionally with progress and a partial result for retries."""
    values: dict[str, Any] = {"heartbeat_at": func.now(), "updated_at": func.now()}
    if done is not None:
        values["progress_done"] = done
    if total is not None:
        values["progress_total"] = total
    if partial is not None:
        values["result"] = partial
    session.execute(
        update(m.Job).where(m.Job.id == job_id, m.Job.state == "running").values(**values)
    )
    session.commit()


def complete(session: Session, job_id: uuid.UUID, result: dict[str, Any] | None = None) -> None:
    session.execute(
        update(m.Job)
        .where(m.Job.id == job_id)
        .values(
            state="done",
            result=result or {},
            finished_at=func.now(),
            updated_at=func.now(),
            heartbeat_at=None,
        )
    )
    session.commit()


def retry_delay(attempts: int) -> timedelta:
    """30 s, 60 s, 120 s, ... after the first, second, third failure."""
    factor: int = 2 ** max(0, attempts - 1)
    return BASE_RETRY_DELAY * factor


def fail(session: Session, job_id: uuid.UUID, error: str) -> str:
    """Record a failure. Returns the new state: ``queued`` (will retry) or ``dead``."""
    job = session.get(m.Job, job_id, populate_existing=True, with_for_update=True)
    if job is None:
        raise KeyError(job_id)
    job.error = error[-4000:]
    job.heartbeat_at = None
    if job.attempts < job.max_attempts:
        job.state = "queued"
        job.run_after = _now(session) + retry_delay(job.attempts)
    else:
        job.state = "dead"
        job.finished_at = _now(session)
    state = job.state
    session.commit()
    return state


def recover_stuck(session: Session, older_than: timedelta = STUCK_AFTER) -> list[uuid.UUID]:
    """Fail running jobs whose heartbeat stopped; they retry or die like any failure."""
    cutoff = _now(session) - older_than
    stuck = session.scalars(
        select(m.Job.id)
        .where(m.Job.state == "running", m.Job.heartbeat_at < cutoff)
        .with_for_update(skip_locked=True)
    ).all()
    minutes = older_than.total_seconds() / 60
    for job_id in stuck:
        fail(session, job_id, f"worker stopped: no heartbeat for {minutes:g} min")
    session.commit()
    return list(stuck)
