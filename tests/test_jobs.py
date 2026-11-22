"""Job queue and worker: claim, retry, crash recovery, no double claims, review batches.

Needs Postgres (CI has it); skipped unless DATABASE_URL is set. Recreates the schema.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from petlens.config import Settings
from petlens.db import migrate
from petlens.db import models as m
from petlens.jobs import queue
from petlens.jobs.handlers import JobContext, batch_files
from petlens.jobs.worker import Worker
from test_api import _analyzer, _cat_photo

DATABASE_URL = os.environ.get("DATABASE_URL")
pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set"),
]


@pytest.fixture
def engine() -> Iterator[Engine]:
    assert DATABASE_URL is not None
    eng = create_engine(DATABASE_URL)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    migrate.upgrade(DATABASE_URL)
    yield eng
    eng.dispose()


@pytest.fixture
def sessions(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


def _state(sessions: sessionmaker[Session], job_id: uuid.UUID) -> m.Job:
    with sessions() as s:
        job = s.get(m.Job, job_id)
        assert job is not None
        return job


def test_claim_takes_oldest_runnable_job_once(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        first = queue.enqueue(s, "a").id
        second = queue.enqueue(s, "b").id
        s.commit()
        a = queue.claim(s)
        assert a is not None and a.id == first
        assert (a.state, a.attempts) == ("running", 1)
        assert a.heartbeat_at is not None
        b = queue.claim(s, ["b"])
        assert b is not None and b.id == second
        assert queue.claim(s) is None


def test_no_double_claims_under_concurrency(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        ids = {queue.enqueue(s, "work").id for _ in range(40)}
        s.commit()
    claimed: list[uuid.UUID] = []
    lock = threading.Lock()
    start = threading.Barrier(6)

    def grab() -> None:
        start.wait()
        with sessions() as s:
            while (job := queue.claim(s)) is not None:
                with lock:
                    claimed.append(job.id)

    threads = [threading.Thread(target=grab) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(claimed) == len(set(claimed)) == 40
    assert set(claimed) == ids


def test_failures_retry_with_backoff_then_die(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        job_id = queue.enqueue(s, "flaky", max_attempts=2).id
        s.commit()
        assert queue.claim(s) is not None
        assert queue.fail(s, job_id, "boom") == "queued"
    job = _state(sessions, job_id)
    assert job.error == "boom"
    with sessions() as s:
        assert queue.claim(s) is None  # waiting out the 30 s retry delay
        s.execute(text("UPDATE job SET run_after = now()"))
        s.commit()
        again = queue.claim(s)
        assert again is not None and again.attempts == 2
        assert queue.fail(s, job_id, "boom again") == "dead"
    job = _state(sessions, job_id)
    assert job.state == "dead" and job.finished_at is not None
    assert queue.retry_delay(1) == timedelta(seconds=30)
    assert queue.retry_delay(3) == timedelta(seconds=120)


def test_stuck_jobs_are_recovered(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        stuck = queue.enqueue(s, "slow").id
        alive = queue.enqueue(s, "slow").id
        s.commit()
        queue.claim(s)
        queue.claim(s)
        # The first worker "crashed" 20 minutes ago; the second is still beating.
        s.execute(
            text("UPDATE job SET heartbeat_at = now() - interval '20 minutes' WHERE id = :id"),
            {"id": stuck},
        )
        s.commit()
        assert queue.recover_stuck(s, timedelta(minutes=10)) == [stuck]
    job = _state(sessions, stuck)
    assert job.state == "queued" and "no heartbeat" in (job.error or "")
    assert _state(sessions, alive).state == "running"


def test_worker_records_success_and_failure(engine: Engine, tmp_path: Path) -> None:
    assert DATABASE_URL is not None
    calls: list[str] = []

    def ok(ctx: JobContext, job: m.Job) -> dict[str, Any]:
        ctx.progress(1, 2, {"half": True})
        calls.append("ok")
        return {"answer": 42}

    def broken(ctx: JobContext, job: m.Job) -> dict[str, Any]:
        raise RuntimeError("disk full")

    settings = Settings(database_url=DATABASE_URL, data_dir=tmp_path)
    worker = Worker(settings, handlers={"ok": ok, "broken": broken}, heartbeat_seconds=0.05)
    with worker.sessions() as s:
        good = queue.enqueue(s, "ok").id
        bad = queue.enqueue(s, "broken", max_attempts=1).id
        s.commit()
    assert worker.run(max_jobs=2) == 2
    done, dead = _state(worker.sessions, good), _state(worker.sessions, bad)
    assert (done.state, done.result, done.progress_done) == ("done", {"answer": 42}, 1)
    assert dead.state == "dead" and "disk full" in (dead.error or "")
    assert worker.run_one() is None


def test_review_batch_end_to_end(engine: Engine, tmp_path: Path) -> None:
    assert DATABASE_URL is not None
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(_cat_photo())
    (folder / "b.jpg").write_bytes(_cat_photo((320, 240)))
    (folder / "broken.png").write_bytes(b"not really a png")
    (folder / "notes.txt").write_text("ignored")
    assert [p.name for p in batch_files(folder)] == ["a.jpg", "b.jpg", "broken.png"]

    analyzer = _analyzer()
    settings = Settings(database_url=DATABASE_URL, data_dir=tmp_path / "data")
    worker = Worker(settings, analyzer=lambda: analyzer)
    with worker.sessions() as s:
        job_id = queue.enqueue(s, "review_batch", {"folder": str(folder)}, total=3).id
        s.commit()
    worker.run(max_jobs=1)
    job = _state(worker.sessions, job_id)
    assert job.state == "done", job.error
    assert (job.progress_done, job.progress_total) == (3, 3)
    assert set(job.result["uploads"]) == {"a.jpg", "b.jpg"}
    assert "broken.png" in job.result["rejected"]
    with engine.connect() as conn:
        kinds = conn.execute(text("SELECT kind FROM upload")).scalars().all()
        dead = conn.execute(text("SELECT source, filename FROM dead_letter")).one()
        labels = conn.execute(text("SELECT label FROM prediction")).scalars().all()
    assert kinds == ["review", "review"]
    assert tuple(dead) == ("worker", "broken.png")
    assert labels == ["ragdoll", "ragdoll"]


def test_review_batch_retry_skips_finished_files(engine: Engine, tmp_path: Path) -> None:
    assert DATABASE_URL is not None
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(_cat_photo())
    (folder / "b.jpg").write_bytes(_cat_photo())
    analyzer = _analyzer()
    settings = Settings(database_url=DATABASE_URL, data_dir=tmp_path / "data")
    worker = Worker(settings, analyzer=lambda: analyzer)
    with worker.sessions() as s:
        job = queue.enqueue(s, "review_batch", {"folder": str(folder)})
        # A previous attempt stored a.jpg before the worker crashed.
        job.result = {"uploads": {"a.jpg": "earlier"}, "rejected": {}}
        s.commit()
    worker.run(max_jobs=1)
    done = _state(worker.sessions, job.id)
    assert done.result["uploads"]["a.jpg"] == "earlier"
    with engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM upload")).scalar() == 1


def test_review_batch_limit(tmp_path: Path) -> None:
    for i in range(201):
        (tmp_path / f"{i:03}.jpg").write_bytes(b"")
    with pytest.raises(ValueError, match="at most 200"):
        batch_files(tmp_path)
