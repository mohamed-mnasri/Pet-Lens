"""The worker: claim a job, run its handler, record the outcome, repeat.

One worker process is enough on one PC (``make worker``). It also recovers jobs left
running by a crashed worker, and stops cleanly on Ctrl+C after the current job.
"""

from __future__ import annotations

import logging
import signal
import threading
import traceback
from collections.abc import Callable
from datetime import timedelta
from functools import lru_cache
from types import FrameType
from typing import Any

from sqlalchemy.orm import sessionmaker

from petlens.config import Settings
from petlens.db import models as m
from petlens.db.session import get_engine
from petlens.jobs import queue
from petlens.jobs.handlers import HANDLERS, Handler, JobContext
from petlens.vision.pipeline import Analyzer

log = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        settings: Settings,
        analyzer: Callable[[], Analyzer] | None = None,
        handlers: dict[str, Handler] | None = None,
        poll_seconds: float = 2.0,
        stuck_after: timedelta = queue.STUCK_AFTER,
        heartbeat_seconds: float = 30.0,
    ) -> None:
        self.settings = settings
        self.sessions = sessionmaker(get_engine(settings.database_url), expire_on_commit=False)
        self.handlers = handlers or HANDLERS
        self.poll_seconds = poll_seconds
        self.stuck_after = stuck_after
        self.heartbeat_seconds = heartbeat_seconds
        self.stopping = threading.Event()

        @lru_cache(maxsize=1)
        def load() -> Analyzer:
            from petlens.vision.load import load_analyzer

            return load_analyzer(settings, settings.detector_size)

        self.analyzer = analyzer or load

    def run_one(self) -> m.Job | None:
        """Recover stuck jobs, then claim and run one job. Returns it, or None if idle."""
        with self.sessions() as session:
            for job_id in queue.recover_stuck(session, self.stuck_after):
                log.warning("job %s had no heartbeat; requeued or marked dead", job_id)
            job = queue.claim(session, list(self.handlers))
        if job is None:
            return None
        log.info("running job %s (%s), attempt %d", job.id, job.kind, job.attempts)

        def progress(done: int, total: int | None, partial: dict[str, Any] | None = None) -> None:
            with self.sessions() as s:
                queue.heartbeat(s, job.id, done, total, partial)

        # A separate heartbeat keeps long steps (one slow file, model loading) from
        # looking like a crash.
        finished = threading.Event()

        def beat() -> None:
            while not finished.wait(self.heartbeat_seconds):
                with self.sessions() as s:
                    queue.heartbeat(s, job.id)

        beater = threading.Thread(target=beat, name=f"heartbeat-{job.id}", daemon=True)
        beater.start()
        ctx = JobContext(self.settings, self.sessions, self.analyzer, progress)
        try:
            result = self.handlers[job.kind](ctx, job)
        except Exception as exc:  # any failure is recorded on the job, never lost
            with self.sessions() as session:
                state = queue.fail(session, job.id, f"{exc}\n{traceback.format_exc()}")
            log.error("job %s failed (%s): %s", job.id, state, exc)
        else:
            with self.sessions() as session:
                queue.complete(session, job.id, result)
            log.info("job %s done", job.id)
        finally:
            finished.set()
            beater.join()
        return job

    def run(self, max_jobs: int | None = None) -> int:
        """Loop until stopped (or ``max_jobs`` have run). Returns how many ran."""
        ran = 0
        while not self.stopping.is_set() and (max_jobs is None or ran < max_jobs):
            if self.run_one() is None:
                self.stopping.wait(self.poll_seconds)
            else:
                ran += 1
        return ran

    def install_signal_handlers(self) -> None:
        def stop(signum: int, _: FrameType | None) -> None:
            log.info("stopping after the current job (signal %d)", signum)
            self.stopping.set()

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
