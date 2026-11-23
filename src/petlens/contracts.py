"""Pydantic contracts for the operational tables (SPEC section 9).

Each contract mirrors one table in ``petlens.db.models``. The allowed values below are
shared with the database check constraints, so the API and Postgres agree on them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, get_args
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

UploadKind = Literal["photo", "video", "live", "review"]
UploadStatus = Literal["received", "stored", "processing", "done", "rejected", "failed"]
Species = Literal["cat", "dog"]
Outcome = Literal["accept", "no_clear_breed", "review"]
LabelOrigin = Literal["oxford", "commons", "me", "vlm", "model"]
LabelAction = Literal["propose", "accept", "reject", "fix", "skip"]
JobState = Literal["queued", "running", "done", "failed", "dead"]
DeadLetterSource = Literal["api", "worker"]

EMBEDDING_DIM = 384  # DINOv2 ViT-S/14


def values(literal: Any) -> tuple[str, ...]:
    """The allowed strings of a Literal type, for check constraints."""
    return tuple(get_args(literal))


class Contract(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="forbid")


class Upload(Contract):
    id: UUID
    kind: UploadKind
    original_name: str = Field(max_length=255)
    mime: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int | None = None
    height: int | None = None
    duration_s: float | None = Field(default=None, ge=0)
    status: UploadStatus
    note: str | None = None
    created_at: datetime
    updated_at: datetime


class Frame(Contract):
    id: UUID
    upload_id: UUID
    frame_idx: int = Field(ge=0)
    seconds: float = Field(ge=0)
    image_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    size_bytes: int = Field(gt=0)
    faces_blurred: int = Field(ge=0)
    created_at: datetime


class Detection(Contract):
    id: UUID
    frame_id: UUID
    species: Species
    score: float = Field(ge=0, le=1)
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    crop_path: str | None = None
    embed_model: str | None = None
    group_idx: int | None = Field(default=None, ge=0)
    created_at: datetime


class TopK(BaseModel):
    label: str
    p: float = Field(ge=0, le=1)


class Prediction(Contract):
    id: UUID
    upload_id: UUID
    detection_id: UUID | None = None
    group_idx: int | None = Field(default=None, ge=0)
    model_version: str
    species: Species
    top3: list[TopK] = Field(max_length=3)
    knn_distance: float
    outcome: Outcome
    label: str | None = None
    also: str | None = None
    confidence: float = Field(ge=0, le=1)
    created_at: datetime


class LabelEvent(Contract):
    id: int
    image_id: str | None = None
    detection_id: UUID | None = None
    species: Species
    proposed_label: str
    origin: LabelOrigin
    action: LabelAction
    reviewer: str | None = None
    review_ms: int | None = Field(default=None, ge=0)
    created_at: datetime


class Job(Contract):
    id: UUID
    kind: str
    payload: dict[str, Any]
    state: JobState
    attempts: int = Field(ge=0)
    max_attempts: int = Field(ge=1)
    error: str | None = None
    result: dict[str, Any]
    progress_done: int = Field(ge=0)
    progress_total: int | None = Field(default=None, ge=0)
    heartbeat_at: datetime | None = None
    run_after: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class DatasetVersion(Contract):
    version: str
    manifest_hash: str
    counts: dict[str, Any]
    created_at: datetime


class ModelRegistry(Contract):
    id: UUID
    run_id: str
    weights_hash: str
    dataset_version: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    champion: bool
    created_at: datetime


class DeadLetter(Contract):
    id: int
    source: DeadLetterSource
    reason: str
    filename: str | None = None
    size_bytes: int | None = None
    payload: dict[str, Any]
    created_at: datetime


class TrainImage(Contract):
    image_id: str
    dataset_version: str
    species: Species
    breed: str
    source: str
    source_ref: str | None = None
    licence: str | None = None
    attribution: str | None = None
    embed_model: str


class Evidence(BaseModel):
    """One of the most similar training photos for a suggested breed."""

    image_id: str
    breed: str
    similarity: float
    url: str
    source: str
    source_ref: str | None = None
    licence: str | None = None
    attribution: str | None = None
