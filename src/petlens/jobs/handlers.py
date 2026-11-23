"""What each job kind does. Handlers report progress through ``JobContext.progress``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from petlens.api import ingest, service
from petlens.config import Settings
from petlens.db import models as m
from petlens.review.cards import cards_for_upload
from petlens.vision.pipeline import Analyzer

REVIEW_BATCH_MAX_FILES = 200
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v"}


@dataclass
class JobContext:
    settings: Settings
    sessions: sessionmaker[Session]
    analyzer: Callable[[], Analyzer]
    # progress(done, total, partial result): also saved so a retry can skip finished work
    progress: Callable[[int, int | None, dict[str, Any] | None], None]


Handler = Callable[[JobContext, m.Job], dict[str, Any]]


def batch_files(folder: Path) -> list[Path]:
    """Photos and clips directly inside ``folder`` (not subfolders), sorted by name."""
    if not folder.is_dir():
        raise ValueError(f"not a folder: {folder}")
    files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES | VIDEO_SUFFIXES
    )
    if len(files) > REVIEW_BATCH_MAX_FILES:
        raise ValueError(
            f"{len(files)} files; a review batch takes at most {REVIEW_BATCH_MAX_FILES}"
        )
    return files


def review_batch(ctx: JobContext, job: m.Job) -> dict[str, Any]:
    """Predict every photo and clip in a folder. Each file is its own transaction, so a
    retry after a crash skips files already stored and a bad file doesn't stop the batch."""
    files = batch_files(Path(job.payload["folder"]))
    analyzer = ctx.analyzer()
    done_names: dict[str, str] = dict(job.result.get("uploads", {}))
    rejected: dict[str, str] = dict(job.result.get("rejected", {}))
    ctx.progress(len(done_names) + len(rejected), len(files), None)
    for path in files:
        if path.name in done_names or path.name in rejected:
            continue
        is_video = path.suffix.lower() in VIDEO_SUFFIXES
        handler = service.process_video if is_video else service.process_photo
        data = path.read_bytes()
        with ctx.sessions() as session:
            try:
                upload_id = handler(session, analyzer, ctx.settings, data, path.name, "review")
                cards_for_upload(session, upload_id)  # swipe cards for each animal
                session.commit()
                done_names[path.name] = str(upload_id)
            except ingest.RejectedUploadError as exc:
                session.rollback()
                service.record_rejection(
                    session, exc.reason, path.name, len(data), "worker", job_id=str(job.id)
                )
                session.commit()
                rejected[path.name] = exc.reason
        partial = {"uploads": done_names, "rejected": rejected}
        ctx.progress(len(done_names) + len(rejected), len(files), partial)
    return {"uploads": done_names, "rejected": rejected}


def commons_screening(ctx: JobContext, job: m.Job) -> dict[str, Any]:
    """Re-run Commons screening (the same as ``make screen``)."""
    from petlens.models.__main__ import main as models_main

    ctx.progress(0, 1, None)
    models_main(["screen"])
    ctx.progress(1, 1, None)
    return {"stage": "screen"}


HANDLERS: dict[str, Handler] = {
    "review_batch": review_batch,
    "commons_screening": commons_screening,
}
