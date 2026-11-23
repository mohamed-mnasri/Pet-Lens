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
LabelAction = Literal["propose", "accept", "reject", "fix", "skip", "unsure"]
ReviewQueue = Literal["commons", "batch", "audit", "check"]
CardState = Literal["pending", "done"]
JobState = Literal["queued", "running", "done", "failed", "dead"]
DeadLetterSource = Literal["api", "worker"]
VlmStatus = Literal["ok", "parse_failed"]
PetStatus = Literal["candidate", "named", "merged"]
IdentityAction = Literal["assign", "new_pet", "confirm", "move", "name", "merge", "split"]
IdentityOrigin = Literal["model", "me"]
VlmKind = Literal["photo", "drawing", "toy", "other"]

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
    pet_id: UUID | None = None
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
    review_card_id: UUID | None = None
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
    gate: dict[str, Any] | None = None
    promoted_at: datetime | None = None
    created_at: datetime


class VlmTag(Contract):
    id: int
    image_id: str
    prompt_version: str
    model_tag: str
    model_digest: str
    status: VlmStatus
    kind: VlmKind | None = None
    whole_animal: bool | None = None
    kitten: bool | None = None
    blurry: bool | None = None
    several_animals: bool | None = None
    attempts: int
    seconds: float
    raw: str
    created_at: datetime


class Pet(Contract):
    id: UUID
    species: Species
    name: str | None = None
    status: PetStatus
    merged_into: UUID | None = None
    created_at: datetime


class IdentityEvent(Contract):
    id: int
    pet_id: UUID
    detection_id: UUID | None = None
    other_pet_id: UUID | None = None
    action: IdentityAction
    origin: IdentityOrigin
    value: str | None = None
    distance: float | None = None
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


class ReviewCard(Contract):
    """One swipe card. ``model_label`` and ``truth`` stay on the server (blind audit and
    reviewer check); the page gets ``CardView`` instead."""

    id: UUID
    ref: str
    queue: ReviewQueue
    image_id: str | None = None
    detection_id: UUID | None = None
    upload_id: UUID | None = None
    image_url: str
    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    species: Species
    suggested: str | None = None
    options: list[str]
    model_label: str | None = None
    truth: str | None = None
    priority: int
    state: CardState
    created_at: datetime
    answered_at: datetime | None = None


class CardView(BaseModel):
    id: UUID
    queue: ReviewQueue
    image_url: str
    box: tuple[float, float, float, float] | None = None
    species: Species
    suggested: str | None  # None on blind audit cards: pick from the options instead
    question: str
    options: list[str]
    references: list[Evidence]
    pending: int


ReviewAnswer = Literal["correct", "wrong", "unsure", "skip"]


class ReviewIn(BaseModel):
    answer: ReviewAnswer
    label: str | None = Field(default=None, max_length=64)  # for "wrong" and blind cards
    review_ms: int | None = Field(default=None, ge=0, le=3_600_000)


class ReviewStats(BaseModel):
    pending: dict[str, int]
    answered: int
    checks_answered: int
    checks_correct: int
    reviewer_accuracy: float | None
