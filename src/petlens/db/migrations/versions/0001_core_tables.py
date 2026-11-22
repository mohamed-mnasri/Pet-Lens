"""Core tables: uploads, frames, detections, predictions, label events, jobs,
dataset versions, model registry and dead letters.

Revision ID: 0001
Revises:
Create Date: 2026-11-22 21:14:49.964438
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


APPEND_ONLY = """
CREATE FUNCTION label_event_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'label_event is append-only';
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER label_event_append_only
    BEFORE UPDATE OR DELETE ON label_event
    FOR EACH ROW EXECUTE FUNCTION label_event_append_only();
"""


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "dataset_version",
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("manifest_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "counts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("version", name=op.f("pk_dataset_version")),
    )
    op.create_table(
        "dead_letter",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("source IN ('api', 'worker')", name=op.f("ck_dead_letter_source")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dead_letter")),
    )
    op.create_table(
        "job",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("state", sa.String(length=16), server_default="queued", nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="3", nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "run_after", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "state IN ('queued', 'running', 'done', 'failed', 'dead')", name=op.f("ck_job_state")
        ),
        sa.CheckConstraint("attempts >= 0 AND max_attempts >= 1", name=op.f("ck_job_attempts")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_job")),
    )
    op.create_index("ix_job_state_run_after", "job", ["state", "run_after"], unique=False)
    op.create_table(
        "upload",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("original_name", sa.String(length=255), nullable=False),
        sa.Column("mime", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("duration_s", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=16), server_default="received", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('photo', 'video', 'live', 'review')", name=op.f("ck_upload_kind")
        ),
        sa.CheckConstraint(
            "status IN ('received', 'stored', 'processing', 'done', 'rejected', 'failed')",
            name=op.f("ck_upload_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_upload")),
    )
    op.create_index(op.f("ix_upload_sha256"), "upload", ["sha256"], unique=False)
    op.create_table(
        "frame",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("upload_id", sa.UUID(), nullable=False),
        sa.Column("frame_idx", sa.Integer(), nullable=False),
        sa.Column("seconds", sa.Float(), server_default="0", nullable=False),
        sa.Column("image_id", sa.String(length=64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("faces_blurred", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["upload_id"], ["upload.id"], name=op.f("fk_frame_upload_id_upload"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_frame")),
        sa.UniqueConstraint("upload_id", "frame_idx", name="upload_frame"),
    )
    op.create_index(op.f("ix_frame_image_id"), "frame", ["image_id"], unique=False)
    op.create_table(
        "model_registry",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("weights_hash", sa.String(length=64), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("champion", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["dataset_version"],
            ["dataset_version.version"],
            name=op.f("fk_model_registry_dataset_version_dataset_version"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_registry")),
        sa.UniqueConstraint("run_id", name=op.f("uq_model_registry_run_id")),
    )
    op.create_index(
        "uq_model_registry_one_champion",
        "model_registry",
        ["champion"],
        unique=True,
        postgresql_where=sa.text("champion"),
    )
    op.create_table(
        "detection",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("frame_id", sa.UUID(), nullable=False),
        sa.Column("species", sa.String(length=8), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("x0", sa.Float(), nullable=False),
        sa.Column("y0", sa.Float(), nullable=False),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("crop_path", sa.Text(), nullable=True),
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=True),
        sa.Column("embed_model", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("species IN ('cat', 'dog')", name=op.f("ck_detection_species")),
        sa.CheckConstraint("0 <= x0 AND x0 < x1 AND x1 <= 1", name=op.f("ck_detection_box_x")),
        sa.CheckConstraint("0 <= y0 AND y0 < y1 AND y1 <= 1", name=op.f("ck_detection_box_y")),
        sa.ForeignKeyConstraint(
            ["frame_id"], ["frame.id"], name=op.f("fk_detection_frame_id_frame"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_detection")),
    )
    op.create_index(op.f("ix_detection_frame_id"), "detection", ["frame_id"], unique=False)
    op.create_table(
        "label_event",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("image_id", sa.String(length=64), nullable=True),
        sa.Column("detection_id", sa.UUID(), nullable=True),
        sa.Column("species", sa.String(length=8), nullable=False),
        sa.Column("proposed_label", sa.String(length=64), nullable=False),
        sa.Column("origin", sa.String(length=16), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("reviewer", sa.String(length=64), nullable=True),
        sa.Column("review_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "action IN ('propose', 'accept', 'reject', 'fix', 'skip')",
            name=op.f("ck_label_event_action"),
        ),
        sa.CheckConstraint(
            "origin IN ('oxford', 'commons', 'me', 'vlm', 'model')",
            name=op.f("ck_label_event_origin"),
        ),
        sa.CheckConstraint("species IN ('cat', 'dog')", name=op.f("ck_label_event_species")),
        sa.CheckConstraint(
            "image_id IS NOT NULL OR detection_id IS NOT NULL", name=op.f("ck_label_event_target")
        ),
        sa.ForeignKeyConstraint(
            ["detection_id"],
            ["detection.id"],
            name=op.f("fk_label_event_detection_id_detection"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_label_event")),
    )
    op.create_index(
        op.f("ix_label_event_detection_id"), "label_event", ["detection_id"], unique=False
    )
    op.create_index(op.f("ix_label_event_image_id"), "label_event", ["image_id"], unique=False)
    op.create_table(
        "prediction",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("upload_id", sa.UUID(), nullable=False),
        sa.Column("detection_id", sa.UUID(), nullable=True),
        sa.Column("group_idx", sa.Integer(), nullable=True),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("species", sa.String(length=8), nullable=False),
        sa.Column("top3", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("knn_distance", sa.Float(), nullable=False),
        sa.Column("outcome", sa.String(length=16), nullable=False),
        sa.Column("label", sa.String(length=64), nullable=True),
        sa.Column("also", sa.String(length=64), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "outcome IN ('accept', 'no_clear_breed', 'review')", name=op.f("ck_prediction_outcome")
        ),
        sa.CheckConstraint("species IN ('cat', 'dog')", name=op.f("ck_prediction_species")),
        sa.CheckConstraint(
            "detection_id IS NOT NULL OR group_idx IS NOT NULL", name=op.f("ck_prediction_target")
        ),
        sa.ForeignKeyConstraint(
            ["detection_id"],
            ["detection.id"],
            name=op.f("fk_prediction_detection_id_detection"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["upload_id"],
            ["upload.id"],
            name=op.f("fk_prediction_upload_id_upload"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_prediction")),
    )
    op.create_index(
        op.f("ix_prediction_detection_id"), "prediction", ["detection_id"], unique=False
    )
    op.create_index(op.f("ix_prediction_upload_id"), "prediction", ["upload_id"], unique=False)
    op.execute(APPEND_ONLY)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS label_event_append_only ON label_event")
    op.execute("DROP FUNCTION IF EXISTS label_event_append_only()")
    op.drop_index(op.f("ix_prediction_upload_id"), table_name="prediction")
    op.drop_index(op.f("ix_prediction_detection_id"), table_name="prediction")
    op.drop_table("prediction")
    op.drop_index(op.f("ix_label_event_image_id"), table_name="label_event")
    op.drop_index(op.f("ix_label_event_detection_id"), table_name="label_event")
    op.drop_table("label_event")
    op.drop_index(op.f("ix_detection_frame_id"), table_name="detection")
    op.drop_table("detection")
    op.drop_index(
        "uq_model_registry_one_champion",
        table_name="model_registry",
        postgresql_where=sa.text("champion"),
    )
    op.drop_table("model_registry")
    op.drop_index(op.f("ix_frame_image_id"), table_name="frame")
    op.drop_table("frame")
    op.drop_index(op.f("ix_upload_sha256"), table_name="upload")
    op.drop_table("upload")
    op.drop_index("ix_job_state_run_after", table_name="job")
    op.drop_table("job")
    op.drop_table("dead_letter")
    op.drop_table("dataset_version")
