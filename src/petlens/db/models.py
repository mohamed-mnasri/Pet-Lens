"""SQLAlchemy models for the operational tables (SPEC section 9).

``vlm_tag`` (Step 13) and the pet identity tables (Step 15) arrive with their steps.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from petlens import contracts as c

NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


def _in(column: str, literal: Any, name: str | None = None) -> CheckConstraint:
    allowed = ", ".join(f"'{v}'" for v in c.values(literal))
    return CheckConstraint(f"{column} IN ({allowed})", name=name or column)


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


def _updated() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Upload(Base):
    __tablename__ = "upload"
    __table_args__ = (_in("kind", c.UploadKind), _in("status", c.UploadStatus))

    id: Mapped[uuid.UUID] = _uuid_pk()
    kind: Mapped[str] = mapped_column(String(16))
    original_name: Mapped[str] = mapped_column(String(255))
    mime: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    duration_s: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), server_default="received")
    note: Mapped[str | None] = mapped_column(Text)  # e.g. why clip grouping was ambiguous
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()


class Frame(Base):
    __tablename__ = "frame"
    __table_args__ = (UniqueConstraint("upload_id", "frame_idx", name="upload_frame"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    upload_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("upload.id", ondelete="CASCADE"))
    frame_idx: Mapped[int] = mapped_column(Integer)
    seconds: Mapped[float] = mapped_column(Float, server_default="0")
    image_id: Mapped[str] = mapped_column(String(64), index=True)
    path: Mapped[str] = mapped_column(Text)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)
    faces_blurred: Mapped[int] = mapped_column(Integer, server_default="0")
    created_at: Mapped[datetime] = _created()


class Detection(Base):
    __tablename__ = "detection"
    __table_args__ = (
        _in("species", c.Species),
        CheckConstraint("0 <= x0 AND x0 < x1 AND x1 <= 1", name="box_x"),
        CheckConstraint("0 <= y0 AND y0 < y1 AND y1 <= 1", name="box_y"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    frame_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("frame.id", ondelete="CASCADE"), index=True
    )
    species: Mapped[str] = mapped_column(String(8))
    score: Mapped[float] = mapped_column(Float)
    x0: Mapped[float] = mapped_column(Float)
    y0: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    crop_path: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(c.EMBEDDING_DIM))
    embed_model: Mapped[str | None] = mapped_column(String(128))
    group_idx: Mapped[int | None] = mapped_column(Integer)  # appearance group in a clip
    created_at: Mapped[datetime] = _created()


class Prediction(Base):
    __tablename__ = "prediction"
    __table_args__ = (
        _in("species", c.Species),
        _in("outcome", c.Outcome),
        CheckConstraint("detection_id IS NOT NULL OR group_idx IS NOT NULL", name="target"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    upload_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("upload.id", ondelete="CASCADE"), index=True
    )
    detection_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("detection.id", ondelete="CASCADE"), index=True
    )
    group_idx: Mapped[int | None] = mapped_column(Integer)
    model_version: Mapped[str] = mapped_column(String(128))
    species: Mapped[str] = mapped_column(String(8))
    top3: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    knn_distance: Mapped[float] = mapped_column(Float)
    outcome: Mapped[str] = mapped_column(String(16))
    label: Mapped[str | None] = mapped_column(String(64))
    also: Mapped[str | None] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = _created()


class LabelEvent(Base):
    """Append-only: a database trigger refuses UPDATE and DELETE."""

    __tablename__ = "label_event"
    __table_args__ = (
        _in("species", c.Species),
        _in("origin", c.LabelOrigin),
        _in("action", c.LabelAction),
        CheckConstraint("image_id IS NOT NULL OR detection_id IS NOT NULL", name="target"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    image_id: Mapped[str | None] = mapped_column(String(64), index=True)
    detection_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("detection.id", ondelete="SET NULL"), index=True
    )
    species: Mapped[str] = mapped_column(String(8))
    proposed_label: Mapped[str] = mapped_column(String(64))
    origin: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(16))
    reviewer: Mapped[str | None] = mapped_column(String(64))
    review_ms: Mapped[int | None] = mapped_column(Integer)
    review_card_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("review_card.id", ondelete="SET NULL"), index=True
    )
    created_at: Mapped[datetime] = _created()


class Job(Base):
    __tablename__ = "job"
    __table_args__ = (
        _in("state", c.JobState),
        CheckConstraint("attempts >= 0 AND max_attempts >= 1", name="attempts"),
        Index("ix_job_state_run_after", "state", "run_after"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    kind: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    state: Mapped[str] = mapped_column(String(16), server_default="queued")
    attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    max_attempts: Mapped[int] = mapped_column(Integer, server_default="3")
    error: Mapped[str | None] = mapped_column(Text)
    result: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    progress_done: Mapped[int] = mapped_column(Integer, server_default="0")
    progress_total: Mapped[int | None] = mapped_column(Integer)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _updated()


class DatasetVersion(Base):
    __tablename__ = "dataset_version"

    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    manifest_hash: Mapped[str] = mapped_column(String(64))
    counts: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = _created()


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (
        # At most one champion at a time.
        Index(
            "uq_model_registry_one_champion",
            "champion",
            unique=True,
            postgresql_where=text("champion"),
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[str] = mapped_column(String(128), unique=True)
    weights_hash: Mapped[str] = mapped_column(String(64))
    dataset_version: Mapped[str] = mapped_column(ForeignKey("dataset_version.version"))
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    champion: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    created_at: Mapped[datetime] = _created()


class DeadLetter(Base):
    __tablename__ = "dead_letter"
    __table_args__ = (_in("source", c.DeadLetterSource),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(Text)
    filename: Mapped[str | None] = mapped_column(String(255))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = _created()


class TrainImage(Base):
    """Training images with their embeddings, for the evidence panel (nearest examples)."""

    __tablename__ = "train_image"
    __table_args__ = (
        _in("species", c.Species),
        Index("ix_train_image_species_breed", "species", "breed"),
    )

    image_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_version: Mapped[str] = mapped_column(String(64))
    species: Mapped[str] = mapped_column(String(8))
    breed: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(32))
    source_ref: Mapped[str | None] = mapped_column(Text)
    licence: Mapped[str | None] = mapped_column(String(64))
    attribution: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(c.EMBEDDING_DIM))
    embed_model: Mapped[str] = mapped_column(String(128))


class ReviewCard(Base):
    """A swipe card. Answers are stored as label events; the card only tracks state."""

    __tablename__ = "review_card"
    __table_args__ = (
        _in("queue", c.ReviewQueue),
        _in("species", c.Species),
        _in("state", c.CardState),
        Index("ix_review_card_state_priority", "state", "priority"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    ref: Mapped[str] = mapped_column(String(200), unique=True)  # queue:image, for idempotent loads
    queue: Mapped[str] = mapped_column(String(16))
    image_id: Mapped[str | None] = mapped_column(String(64), index=True)
    detection_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("detection.id", ondelete="CASCADE")
    )
    upload_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("upload.id", ondelete="CASCADE"))
    image_url: Mapped[str] = mapped_column(Text)
    x0: Mapped[float | None] = mapped_column(Float)
    y0: Mapped[float | None] = mapped_column(Float)
    x1: Mapped[float | None] = mapped_column(Float)
    y1: Mapped[float | None] = mapped_column(Float)
    species: Mapped[str] = mapped_column(String(8))
    suggested: Mapped[str | None] = mapped_column(String(64))
    options: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))
    model_label: Mapped[str | None] = mapped_column(String(64))
    truth: Mapped[str | None] = mapped_column(String(64))
    priority: Mapped[int] = mapped_column(Integer, server_default="1")
    state: Mapped[str] = mapped_column(String(16), server_default="pending")
    created_at: Mapped[datetime] = _created()
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --- evaluation (Step 10): written by ``make eval``, read by the dbt marts ------------


class EvalRun(Base):
    """One evaluation of a training run on its dataset version's test sets."""

    __tablename__ = "eval_run"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[str] = mapped_column(String(128))
    dataset_version: Mapped[str] = mapped_column(String(64))
    embedding_model: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = _created()


class EvalMetric(Base):
    """A headline metric for one species and test set, with a 95% interval when it has one.
    ``param`` holds the coverage for risk-coverage points."""

    __tablename__ = "eval_metric"
    __table_args__ = (
        _in("species", c.Species),
        Index("ix_eval_metric_run", "eval_run_id", "species", "test_set"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_run.id", ondelete="CASCADE")
    )
    species: Mapped[str] = mapped_column(String(8))
    test_set: Mapped[str] = mapped_column(String(32))
    metric: Mapped[str] = mapped_column(String(48))
    param: Mapped[float | None] = mapped_column(Float)
    value: Mapped[float] = mapped_column(Float)
    lo: Mapped[float | None] = mapped_column(Float)
    hi: Mapped[float | None] = mapped_column(Float)
    n: Mapped[int] = mapped_column(Integer)


class EvalClassMetric(Base):
    """Recall and precision per class, with Wilson intervals on recall."""

    __tablename__ = "eval_class_metric"
    __table_args__ = (
        _in("species", c.Species),
        Index("ix_eval_class_metric_run", "eval_run_id", "species", "test_set"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_run.id", ondelete="CASCADE")
    )
    species: Mapped[str] = mapped_column(String(8))
    test_set: Mapped[str] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(String(64))
    n: Mapped[int] = mapped_column(Integer)
    recall: Mapped[float] = mapped_column(Float)
    recall_lo: Mapped[float] = mapped_column(Float)
    recall_hi: Mapped[float] = mapped_column(Float)
    precision: Mapped[float | None] = mapped_column(Float)


class EvalImage(Base):
    """One test image's result: what the head said and what the policy decided."""

    __tablename__ = "eval_image"
    __table_args__ = (
        _in("species", c.Species),
        _in("outcome", c.Outcome),
        Index("ix_eval_image_run", "eval_run_id", "species", "test_set"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("eval_run.id", ondelete="CASCADE")
    )
    image_id: Mapped[str] = mapped_column(String(64))
    species: Mapped[str] = mapped_column(String(8))
    test_set: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(32))
    label_origin: Mapped[str] = mapped_column(String(32))
    true_label: Mapped[str] = mapped_column(String(64))
    known_breed: Mapped[bool] = mapped_column(Boolean)
    predicted_label: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    knn_distance: Mapped[float] = mapped_column(Float)
    outcome: Mapped[str] = mapped_column(String(16))
    decided_label: Mapped[str | None] = mapped_column(String(64))
    top1_correct: Mapped[bool] = mapped_column(Boolean)
    flags: Mapped[list[str]] = mapped_column(JSONB, server_default=text("'[]'::jsonb"))


class DatasetCount(Base):
    """Dataset composition per version (from the manifest), for the composition mart."""

    __tablename__ = "dataset_count"
    __table_args__ = (Index("ix_dataset_count_version", "dataset_version"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    dataset_version: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(32))
    species: Mapped[str] = mapped_column(String(8))
    breed: Mapped[str] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String(32))
    label_origin: Mapped[str] = mapped_column(String(32))
    n: Mapped[int] = mapped_column(Integer)
