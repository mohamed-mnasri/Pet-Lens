"""HTTP API: ``/predict/image``, ``/predict/video``, ``/runs/{id}``, ``/jobs/{id}``.

Prediction runs inline for now; Step 7 moves it onto the job queue. Every accepted
upload lands in Postgres, and every rejected one is logged to ``dead_letter``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker

from petlens import contracts as c
from petlens.api import ingest, service
from petlens.config import Settings, get_settings
from petlens.db import models as m
from petlens.db.session import get_engine
from petlens.vision.pipeline import Analyzer

log = logging.getLogger(__name__)


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
    app = FastAPI(title="PetLens", version="0.1.0")

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

    @app.get("/jobs/{job_id}", response_model=c.Job)
    def get_job(job_id: uuid.UUID, session: DB) -> c.Job:
        job = session.get(m.Job, job_id)
        if job is None:
            raise HTTPException(404, "No job with that id")
        return c.Job.model_validate(job)

    return app
