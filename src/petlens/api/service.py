"""Store an upload and its results: compressed frames, crops and database rows.

Originals are never kept. A photo, or each sampled frame of a clip, is resized to the
storage long edge, analysed, face-blurred and re-encoded as WebP under the byte cap.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from petlens import contracts as c
from petlens.api import ingest
from petlens.config import Settings
from petlens.data.imaging import (
    DEFAULT_POLICY,
    StoragePolicy,
    decode,
    encode_webp,
    resize_long_edge,
    sha256_bytes,
    stored_path,
)
from petlens.db import models as m
from petlens.db.evidence import evidence_for
from petlens.models.policy import Decision
from petlens.pets.match import assign_upload
from petlens.vision.pipeline import Analyzer, ImageResult, analyse_clip, analyse_image
from petlens.vision.video import Frame, sample_frames


class Run(c.Contract):
    """Everything recorded for one upload."""

    upload: c.Upload
    frames: list[c.Frame]
    detections: list[c.Detection]
    predictions: list[c.Prediction]
    # prediction id -> nearest training photos
    evidence: dict[str, list[c.Evidence]] = Field(default_factory=dict)


@dataclass(frozen=True)
class Stored:
    image_id: str
    path: Path
    width: int
    height: int
    nbytes: int


def _write(img: Image.Image, folder: Path, policy: StoragePolicy) -> Stored:
    data = encode_webp(img, policy)
    image_id = sha256_bytes(data)
    path = stored_path(folder, image_id)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    with Image.open(path) as written:
        w, h = written.size
    return Stored(image_id, path, w, h, len(data))


def _crop(
    img: Image.Image, box: tuple[float, float, float, float], margin: float = 0.1
) -> Image.Image:
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    w, h = img.size
    return img.crop(
        (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(w, int(x1 + mx)), min(h, int(y1 + my)))
    )


def _prediction_row(
    upload_id: uuid.UUID,
    species: str,
    d: Decision,
    version: str,
    detection_id: uuid.UUID | None = None,
    group_idx: int | None = None,
) -> m.Prediction:
    return m.Prediction(
        upload_id=upload_id,
        detection_id=detection_id,
        group_idx=group_idx,
        model_version=version,
        species=species,
        top3=[{"label": label, "p": round(float(p), 6)} for label, p in d.top3],
        knn_distance=float(d.knn_distance),
        outcome=d.outcome,
        label=d.label,
        also=d.also,
        confidence=float(d.confidence),
    )


def _store_frame(
    session: Session,
    settings: Settings,
    upload: m.Upload,
    frame_idx: int,
    seconds: float,
    res: ImageResult,
    version: str,
    per_detection_predictions: bool,
    group_of: dict[int, int] | None = None,
) -> None:
    """Store one frame and its detections. ``group_of`` maps animal index to clip group."""
    stored = _write(res.image, settings.uploads_dir / "frames", DEFAULT_POLICY)
    frame = m.Frame(
        upload_id=upload.id,
        frame_idx=frame_idx,
        seconds=seconds,
        image_id=stored.image_id,
        path=str(stored.path),
        width=stored.width,
        height=stored.height,
        size_bytes=stored.nbytes,
        faces_blurred=res.people_blurred,
    )
    session.add(frame)
    session.flush()
    w, h = res.image.size
    crop_policy = StoragePolicy(
        long_edge=settings.crop_long_edge, max_bytes=settings.crop_max_bytes
    )
    for j, animal in enumerate(res.animals):
        crop = resize_long_edge(_crop(res.image, animal.box), settings.crop_long_edge)
        crop_stored = _write(crop, settings.uploads_dir / "crops", crop_policy)
        x0, y0, x1, y1 = animal.box
        emb = [float(v) for v in animal.embedding]
        det = m.Detection(
            frame_id=frame.id,
            species=animal.species,
            score=float(animal.detection_score),
            x0=max(0.0, x0 / w),
            y0=max(0.0, y0 / h),
            x1=min(1.0, x1 / w),
            y1=min(1.0, y1 / h),
            crop_path=str(crop_stored.path),
            embedding=emb if len(emb) == c.EMBEDDING_DIM else None,
            embed_model=version,
            group_idx=(group_of or {}).get(j),
        )
        session.add(det)
        session.flush()
        if per_detection_predictions:
            session.add(
                _prediction_row(upload.id, animal.species, animal.decision, version, det.id)
            )


def process_photo(
    session: Session,
    analyzer: Analyzer,
    settings: Settings,
    data: bytes,
    name: str | None,
    kind: c.UploadKind = "photo",
) -> uuid.UUID:
    ingest.check_size(len(data), "photo", settings)
    ft = ingest.check_type(data, "photo")
    width, height = ingest.check_photo_pixels(data, settings)
    upload = m.Upload(
        kind=kind,
        original_name=ingest.safe_name(name),
        mime=ft.mime,
        size_bytes=len(data),
        sha256=sha256_bytes(data),
        width=width,
        height=height,
        status="received",
    )
    session.add(upload)
    session.flush()
    try:
        img = resize_long_edge(decode(data), DEFAULT_POLICY.long_edge)  # EXIF applied once
    except ValueError as exc:
        raise ingest.RejectedUploadError(str(exc), 422) from exc
    res = analyse_image(analyzer, img)
    _store_frame(session, settings, upload, 0, 0.0, res, analyzer.version, True)
    assign_upload(session, upload.id)  # which of my pets is it (Step 15)
    upload.status = "done"
    return upload.id


def process_video(
    session: Session,
    analyzer: Analyzer,
    settings: Settings,
    data: bytes,
    name: str | None,
    kind: c.UploadKind = "video",
) -> uuid.UUID:
    ingest.check_size(len(data), "video", settings)
    ft = ingest.check_type(data, "video")
    settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    # The inbox file is named by us, never by the uploader, and deleted straight after.
    tmp = settings.inbox_dir / f"{uuid.uuid4().hex}{ft.suffix}"
    try:
        tmp.write_bytes(data)
        seconds, width, height = ingest.probe_video(tmp)
        ingest.check_duration(seconds, settings)
        raw = sample_frames(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    if not raw:
        raise ingest.RejectedUploadError("No frames could be read from the clip", 422)
    upload = m.Upload(
        kind=kind,
        original_name=ingest.safe_name(name),
        mime=ft.mime,
        size_bytes=len(data),
        sha256=sha256_bytes(data),
        width=width,
        height=height,
        duration_s=seconds,
        status="received",
    )
    session.add(upload)
    session.flush()
    frames = [
        Frame(f.index, f.seconds, resize_long_edge(f.image, DEFAULT_POLICY.long_edge)) for f in raw
    ]
    clip = analyse_clip(analyzer, frames)
    group_of: dict[tuple[int, int], int] = {
        spot: gi for gi, g in enumerate(clip.groups) for spot in g.members
    }
    for pos, (f, res) in enumerate(zip(frames, clip.frames, strict=True)):
        in_frame = {j: gi for (p, j), gi in group_of.items() if p == pos}
        _store_frame(
            session,
            settings,
            upload,
            f.index,
            f.seconds,
            res,
            analyzer.version,
            clip.ambiguous,
            in_frame,
        )
    for i, g in enumerate(clip.groups):
        session.add(
            _prediction_row(upload.id, g.species, g.decision, analyzer.version, group_idx=i)
        )
    if clip.ambiguous:
        upload.note = f"grouping is ambiguous, per-frame results instead: {clip.reason}"
    session.flush()
    assign_upload(session, upload.id)
    upload.status = "done"
    return upload.id


def record_rejection(
    session: Session,
    reason: str,
    name: str | None,
    nbytes: int | None,
    source: c.DeadLetterSource = "api",
    **payload: Any,
) -> None:
    session.add(
        m.DeadLetter(
            source=source,
            reason=reason,
            filename=ingest.safe_name(name),
            size_bytes=nbytes,
            payload=payload,
        )
    )


def load_run(session: Session, upload_id: uuid.UUID) -> Run | None:
    upload = session.get(m.Upload, upload_id)
    if upload is None:
        return None
    frames = session.scalars(
        select(m.Frame).where(m.Frame.upload_id == upload_id).order_by(m.Frame.frame_idx)
    ).all()
    detections = session.scalars(
        select(m.Detection)
        .join(m.Frame)
        .where(m.Frame.upload_id == upload_id)
        .order_by(m.Frame.frame_idx, m.Detection.created_at)
    ).all()
    predictions = session.scalars(
        select(m.Prediction)
        .where(m.Prediction.upload_id == upload_id)
        .order_by(m.Prediction.group_idx, m.Prediction.created_at)
    ).all()
    return Run(
        upload=c.Upload.model_validate(upload),
        frames=[c.Frame.model_validate(f) for f in frames],
        detections=[c.Detection.model_validate(d) for d in detections],
        predictions=[c.Prediction.model_validate(p) for p in predictions],
        evidence={str(p.id): evidence_for(session, p) for p in predictions},
    )
