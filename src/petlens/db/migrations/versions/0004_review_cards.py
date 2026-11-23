"""Swipe review cards, and "unsure" as a label action (Step 9).

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-23 10:15:47.707390
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


OLD_ACTIONS = "'propose', 'accept', 'reject', 'fix', 'skip'"
NEW_ACTIONS = OLD_ACTIONS + ", 'unsure'"


def upgrade() -> None:
    op.create_table(
        "review_card",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("ref", sa.String(length=200), nullable=False),
        sa.Column("queue", sa.String(length=16), nullable=False),
        sa.Column("image_id", sa.String(length=64), nullable=True),
        sa.Column("detection_id", sa.UUID(), nullable=True),
        sa.Column("upload_id", sa.UUID(), nullable=True),
        sa.Column("image_url", sa.Text(), nullable=False),
        sa.Column("x0", sa.Float(), nullable=True),
        sa.Column("y0", sa.Float(), nullable=True),
        sa.Column("x1", sa.Float(), nullable=True),
        sa.Column("y1", sa.Float(), nullable=True),
        sa.Column("species", sa.String(length=8), nullable=False),
        sa.Column("suggested", sa.String(length=64), nullable=True),
        sa.Column(
            "options",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("model_label", sa.String(length=64), nullable=True),
        sa.Column("truth", sa.String(length=64), nullable=True),
        sa.Column("priority", sa.Integer(), server_default="1", nullable=False),
        sa.Column("state", sa.String(length=16), server_default="pending", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "queue IN ('commons', 'batch', 'audit', 'check')", name=op.f("ck_review_card_queue")
        ),
        sa.CheckConstraint("species IN ('cat', 'dog')", name=op.f("ck_review_card_species")),
        sa.CheckConstraint("state IN ('pending', 'done')", name=op.f("ck_review_card_state")),
        sa.ForeignKeyConstraint(
            ["detection_id"],
            ["detection.id"],
            name=op.f("fk_review_card_detection_id_detection"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["upload_id"],
            ["upload.id"],
            name=op.f("fk_review_card_upload_id_upload"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_review_card")),
        sa.UniqueConstraint("ref", name=op.f("uq_review_card_ref")),
    )
    op.create_index(op.f("ix_review_card_image_id"), "review_card", ["image_id"], unique=False)
    op.create_index(
        "ix_review_card_state_priority", "review_card", ["state", "priority"], unique=False
    )
    op.add_column("label_event", sa.Column("review_card_id", sa.UUID(), nullable=True))
    op.create_index(
        op.f("ix_label_event_review_card_id"), "label_event", ["review_card_id"], unique=False
    )
    op.create_foreign_key(
        op.f("fk_label_event_review_card_id_review_card"),
        "label_event",
        "review_card",
        ["review_card_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.drop_constraint(op.f("ck_label_event_action"), "label_event", type_="check")
    op.create_check_constraint(
        op.f("ck_label_event_action"), "label_event", f"action IN ({NEW_ACTIONS})"
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_label_event_action"), "label_event", type_="check")
    op.create_check_constraint(
        op.f("ck_label_event_action"), "label_event", f"action IN ({OLD_ACTIONS})"
    )
    op.drop_constraint(
        op.f("fk_label_event_review_card_id_review_card"), "label_event", type_="foreignkey"
    )
    op.drop_index(op.f("ix_label_event_review_card_id"), table_name="label_event")
    op.drop_column("label_event", "review_card_id")
    op.drop_index("ix_review_card_state_priority", table_name="review_card")
    op.drop_index(op.f("ix_review_card_image_id"), table_name="review_card")
    op.drop_table("review_card")
