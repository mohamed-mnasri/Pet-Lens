"""HTTP API: ``/predict/image``, ``/predict/video``, ``/runs/{id}``, ``/jobs/{id}``.

Prediction runs inline for now; Step 7 moves it onto the job queue. Every accepted
upload lands in Postgres, and every rejected one is logged to ``dead_letter``.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from petlens import contracts as c
from petlens.api import health as model_health_data
from petlens.api import ingest, service
from petlens.config import Settings, get_settings
from petlens.data import own
from petlens.data.catalog import load_breeds
from petlens.data.imaging import decode, stored_path
from petlens.db import models as m
from petlens.db.session import get_engine
from petlens.jobs import queue
from petlens.jobs.handlers import batch_files
from petlens.pets import manage, profiles
from petlens.review import answers
from petlens.vision import live
from petlens.vision.pipeline import Analyzer, analyse_image

log = logging.getLogger(__name__)
HEX64 = re.compile(r"[0-9a-f]{64}")


class PetName(BaseModel):
    name: str


class PetMerge(BaseModel):
    source: uuid.UUID
    target: uuid.UUID


class PetSplit(BaseModel):
    detection_ids: list[uuid.UUID]


class DetectionMove(BaseModel):
    pet_id: uuid.UUID


class ReviewBatch(BaseModel):
    folder: str  # a folder on this PC; the worker reads its photos and clips


def _session(request: Request) -> Iterator[Session]:
    with request.app.state.sessions() as session:
        yield session


DB = Annotated[Session, Depends(_session)]


def create_app(
    settings: Settings | None = None,
    analyzer: Callable[[], Analyzer] | None = None,
    require_models: bool = True,
) -> FastAPI:
    """Build the app. Tests pass their own settings and a fake analyzer. With
    ``require_models=False`` (the demo) the app starts without a trained model and the
    endpoints that need one answer 503."""
    cfg = settings or get_settings()
    factory = sessionmaker(get_engine(cfg.database_url), expire_on_commit=False)

    @lru_cache(maxsize=1)
    def default_analyzer() -> Analyzer:
        from petlens.vision.load import load_analyzer

        return load_analyzer(cfg, cfg.detector_size)

    load = analyzer or default_analyzer
    no_model: list[str] = []

    def get_analyzer() -> Analyzer:
        if no_model:
            raise HTTPException(503, no_model[0])
        try:
            return load()
        except (Exception, SystemExit) as exc:
            if require_models:
                raise
            no_model.append(
                f"No trained model on this machine ({exc}). The demo shows everything "
                "else; `make train` gives it a model."
            )
            raise HTTPException(503, no_model[0]) from exc

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load the models before the first request (about 70 s cold), not during it.
        try:
            get_analyzer()
        except HTTPException as exc:
            log.warning("%s", exc.detail)
        yield

    app = FastAPI(title="PetLens", version="1.0.0", lifespan=lifespan)

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
        data = file.file.read(cfg.photo_max_bytes + 1)  # sync: runs in the thread pool
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
    def review_next(
        session: DB,
        queue: c.ReviewQueue | None = None,
        species: c.Species | None = None,
        breed: str | None = None,
    ) -> c.CardView | None:
        return answers.next_card(session, queue, species, breed)

    @app.get("/review/breeds")
    def review_breeds(session: DB, species: c.Species | None = None) -> dict[str, int]:
        """Pending cards per breed, for the page's one-breed swipe picker (dev tool)."""
        return answers.pending_by_breed(session, species)

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

    # --- pets (Step 15) -------------------------------------------------------------

    def _pet_change(session: Session, change: Callable[[], object]) -> None:
        try:
            change()
        except KeyError as exc:
            raise HTTPException(404, "No such pet or photo") from exc
        except manage.PetError as exc:
            raise HTTPException(409, str(exc)) from exc
        session.commit()

    @app.get("/pets")
    def list_pets(session: DB) -> list[dict[str, object]]:
        return profiles.list_pets(session)

    @app.get("/pets/{pet_id}")
    def get_pet(pet_id: uuid.UUID, session: DB) -> dict[str, object]:
        pet = profiles.pet_detail(session, pet_id)
        if pet is None:
            raise HTTPException(404, "No such pet")
        return pet

    @app.post("/pets/{pet_id}/name")
    def name_pet(pet_id: uuid.UUID, body: PetName, session: DB) -> dict[str, str]:
        _pet_change(session, lambda: manage.name(session, pet_id, body.name))
        return {"status": "ok"}

    @app.post("/pets/merge")
    def merge_pets(body: PetMerge, session: DB) -> dict[str, str]:
        _pet_change(session, lambda: manage.merge(session, body.source, body.target))
        return {"status": "ok"}

    @app.post("/pets/{pet_id}/split")
    def split_pet(pet_id: uuid.UUID, body: PetSplit, session: DB) -> dict[str, str]:
        made: list[str] = []
        _pet_change(
            session,
            lambda: made.append(str(manage.split(session, pet_id, body.detection_ids).id)),
        )
        return {"status": "ok", "pet_id": made[0]}

    @app.post("/detections/{detection_id}/move")
    def move_detection(detection_id: uuid.UUID, body: DetectionMove, session: DB) -> dict[str, str]:
        _pet_change(session, lambda: manage.move(session, detection_id, body.pet_id))
        return {"status": "ok"}

    @app.post("/detections/{detection_id}/confirm")
    def confirm_detection(detection_id: uuid.UUID, session: DB) -> dict[str, str]:
        _pet_change(session, lambda: manage.confirm(session, detection_id))
        return {"status": "ok"}

    # --- live camera (Step 16) --------------------------------------------------------

    live_sessions: dict[str, live.LiveSession] = {}
    page = Path(__file__).parent / "static" / "live.html"

    @app.get("/live", response_class=HTMLResponse)
    def live_page() -> HTMLResponse:
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.post("/live/frame")
    def live_frame(
        session: DB,
        file: Annotated[UploadFile, File()],
        session_id: Annotated[str, Form(alias="session")],
    ) -> dict[str, object]:
        vlm_busy = session.scalar(
            select(m.Job.id).where(m.Job.kind == "vlm_screen", m.Job.state == "running").limit(1)
        )
        if vlm_busy is not None:
            raise HTTPException(409, "The VLM screener is using the GPU. Live view waits for it.")
        data = file.file.read(cfg.photo_max_bytes + 1)  # sync: runs in the thread pool
        ingest.check_size(len(data), "photo", cfg)
        try:
            img = decode(data)
        except ValueError as exc:
            raise HTTPException(422, "That frame isn't an image") from exc
        live.mark_live(cfg.data_dir)
        analyzer = get_analyzer()
        res = analyse_image(analyzer, img)
        key = session_id[:64]
        state = live_sessions.get(key)
        if state is None:
            for old in [
                k for k, v in live_sessions.items() if v.last_seen < time.monotonic() - 120
            ]:
                del live_sessions[old]
            state = live.LiveSession(
                analyzer.policy,
                {s: h.classes for s, h in analyzer.heads.items()},
                analyzer.close_pairs,
            )
            live_sessions[key] = state
        smoothed = state.update(res.animals)
        state.frame_seconds.append(res.seconds)
        if state.should_store(bool(res.animals)):
            service.store_live_frame(session, cfg, res, analyzer.version, len(data))
            session.commit()
        w, h = res.image.size
        catalog = load_breeds()

        def name(label: str | None) -> str | None:
            if label is None:
                return None
            return catalog.get(label).name if label != "domestic_cat" else "Domestic cat"

        return {
            "animals": [
                {
                    "track": a.track,
                    "species": a.species,
                    "box": [a.box[0] / w, a.box[1] / h, a.box[2] / w, a.box[3] / h],
                    "outcome": a.decision.outcome,
                    "label": name(a.decision.label),
                    "also": name(a.decision.also),
                    "confidence": a.decision.confidence,
                    "frames": a.frames,
                    "top3": [{"label": name(lb), "p": p} for lb, p in a.decision.top3],
                }
                for a in smoothed
            ],
            "seconds": res.seconds,
            "fps": state.fps(),
        }

    @app.get("/model-health")
    def model_health(session: DB) -> dict[str, object]:
        """The Model Health page: registry, latest evaluation, confusions, live abstention
        and review throughput, all read from the dbt marts."""
        return model_health_data.model_health(session)

    @app.get("/jobs/{job_id}", response_model=c.Job)
    def get_job(job_id: uuid.UUID, session: DB) -> c.Job:
        job = session.get(m.Job, job_id)
        if job is None:
            raise HTTPException(404, "No job with that id")
        return c.Job.model_validate(job)

    return app
