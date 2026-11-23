"""HTTP API: ``/predict/image``, ``/predict/video``, ``/runs/{id}``, ``/jobs/{id}``.

Prediction runs inline for now; Step 7 moves it onto the job queue. Every accepted
upload lands in Postgres, and every rejected one is logged to ``dead_letter``.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from petlens import contracts as c
from petlens.api import ingest, service
from petlens.config import Settings, get_settings
from petlens.data import own
from petlens.data.catalog import load_breeds
from petlens.data.imaging import stored_path
from petlens.db import models as m
from petlens.db.session import get_engine
from petlens.jobs import queue
from petlens.jobs.handlers import batch_files
from petlens.review import answers
from petlens.vision.pipeline import Analyzer

log = logging.getLogger(__name__)
HEX64 = re.compile(r"[0-9a-f]{64}")


class ReviewBatch(BaseModel):
    folder: str  # a folder on this PC; the worker reads its photos and clips


def _session(request: Request) -> Iterator[Session]:
    with request.app.state.sessions() as session:
        yield session


DB = Annotated[Session, Depends(_session)]


def create_app(
    settings: Settings | None = None, analyzer: Callable[[], Analyzer] | None = None
) -> FastAPI:
    """Build the app. Tests pass their own settings and a fake analyzer."""
    cfg = settings or get_settings()
    factory = sessionmaker(get_engine(cfg.database_url), expire_on_commit=False)

    @lru_cache(maxsize=1)
    def default_analyzer() -> Analyzer:
        from petlens.vision.load import load_analyzer

        return load_analyzer(cfg, cfg.detector_size)

    get_analyzer = analyzer or default_analyzer

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load the models before the first request (about 70 s cold), not during it.
        get_analyzer()
        yield

    app = FastAPI(title="PetLens", version="0.1.0", lifespan=lifespan)

    app.state.sessions = factory

    @app.exception_handler(ingest.RejectedUploadError)
    async def rejected(_: Request, exc: ingest.RejectedUploadError) -> JSONResponse:
        return JSONResponse({"detail": exc.reason}, status_code=exc.status)

    async def predict(session: Session, kind: ingest.Kind, file: UploadFile) -> service.Run:
        limit = cfg.photo_max_bytes if kind == "photo" else cfg.video_max_bytes
        size = file.size
        name = file.filename
        handler = service.process_photo if kind == "photo" else service.process_video
        try:
            if size is not None:
                ingest.check_size(size, kind, cfg)
            data = await file.read(limit + 1)  # never hold more than the limit
            size = size if size is not None else len(data)
            upload_id = handler(session, get_analyzer(), cfg, data, name)
            session.commit()
        except ingest.RejectedUploadError as exc:
            session.rollback()
            service.record_rejection(session, exc.reason, name, size, kind=kind)
            session.commit()
            log.info("rejected %s upload: %s", kind, exc.reason)
            raise
        run = service.load_run(session, upload_id)
        assert run is not None
        return run

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict/image", response_model=service.Run)
    async def predict_image(session: DB, file: Annotated[UploadFile, File()]) -> service.Run:
        return await predict(session, "photo", file)

    @app.post("/predict/video", response_model=service.Run)
    async def predict_video(session: DB, file: Annotated[UploadFile, File()]) -> service.Run:
        return await predict(session, "video", file)

    @app.get("/runs/{upload_id}", response_model=service.Run)
    def get_run(upload_id: uuid.UUID, session: DB) -> service.Run:
        run = service.load_run(session, upload_id)
        if run is None:
            raise HTTPException(404, "No upload with that id")
        return run

    @app.post("/jobs/review-batch", response_model=c.Job, status_code=202)
    def queue_review_batch(body: ReviewBatch, session: DB) -> c.Job:
        try:
            files = batch_files(Path(body.folder))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        job = queue.enqueue(
            session, "review_batch", {"folder": str(Path(body.folder).resolve())}, total=len(files)
        )
        session.commit()
        return c.Job.model_validate(job)

    @app.get("/jobs", response_model=list[c.Job])
    def list_jobs(session: DB, limit: int = 20) -> list[c.Job]:
        jobs = session.scalars(
            select(m.Job).order_by(m.Job.created_at.desc()).limit(min(max(limit, 1), 100))
        )
        return [c.Job.model_validate(j) for j in jobs]

    @app.get("/limits")
    def limits() -> dict[str, float]:
        """Upload limits, so the page can check files before sending them."""
        return {
            "photo_max_bytes": cfg.photo_max_bytes,
            "photo_max_pixels": cfg.photo_max_pixels,
            "video_max_bytes": cfg.video_max_bytes,
            "video_max_seconds": cfg.video_max_seconds,
        }

    breeds = load_breeds()

    @app.get("/breeds")
    def list_breeds() -> list[dict[str, str]]:
        """The breeds the model trains on, for the page's breed picker."""
        return [
            {"id": b.id, "name": b.name, "species": b.species}
            for b in breeds.breeds
            if b.role == "train"
        ]

    @app.post("/training-photos")
    async def add_training_photo(
        breed: Annotated[str, Form()], file: Annotated[UploadFile, File()]
    ) -> dict[str, str | None]:
        """One of my own photos of a known breed, for training (``petlens.data.own``)."""
        if breed not in {b["id"] for b in list_breeds()}:
            raise HTTPException(422, f"Unknown breed: {breed}")
        if file.size is not None:
            ingest.check_size(file.size, "photo", cfg)
        data = await file.read(cfg.photo_max_bytes + 1)
        ingest.check_size(len(data), "photo", cfg)
        ingest.check_type(data, "photo")
        ingest.check_photo_pixels(data, cfg)
        result = own.add_photo(data, breed, cfg.data_dir, breeds, get_analyzer().detector)
        return {"status": result.status, "reason": result.reason, "image_id": result.image_id}

    @app.get("/training-photos")
    def training_photo_counts() -> dict[str, int]:
        return own.counts(cfg.data_dir)

    folders = {
        "train": cfg.processed_dir / "images",
        "frames": cfg.uploads_dir / "frames",
        "crops": cfg.uploads_dir / "crops",
    }

    @app.get("/images/{kind}/{image_id}")
    def image(kind: str, image_id: str) -> FileResponse:
        # Only content-hash names inside known folders: no path from the request is used.
        if kind not in folders or not HEX64.fullmatch(image_id):
            raise HTTPException(404, "No such image")
        path = stored_path(folders[kind], image_id)
        if not path.is_file():
            raise HTTPException(404, "No such image")
        return FileResponse(path, media_type="image/webp")

    @app.get("/review/next", response_model=c.CardView | None)
    def review_next(session: DB, queue: c.ReviewQueue | None = None) -> c.CardView | None:
        return answers.next_card(session, queue)

    @app.post("/review/{card_id}", response_model=c.LabelEvent)
    def review_answer(card_id: uuid.UUID, body: c.ReviewIn, session: DB) -> c.LabelEvent:
        try:
            event = answers.answer(session, card_id, body)
        except KeyError as exc:
            raise HTTPException(404, "No card with that id") from exc
        except answers.ReviewError as exc:
            raise HTTPException(409, str(exc)) from exc
        session.commit()
        return c.LabelEvent.model_validate(event)

    @app.get("/review/stats", response_model=c.ReviewStats)
    def review_stats(session: DB) -> c.ReviewStats:
        return answers.stats(session)

    @app.get("/jobs/{job_id}", response_model=c.Job)
    def get_job(job_id: uuid.UUID, session: DB) -> c.Job:
        job = session.get(m.Job, job_id)
        if job is None:
            raise HTTPException(404, "No job with that id")
        return c.Job.model_validate(job)

    return app
