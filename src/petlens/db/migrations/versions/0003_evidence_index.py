"""Training image index for the evidence panel, and clip groups on detections (Step 8).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-23 00:10:33.150237
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "train_image",
        sa.Column("image_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_version", sa.String(length=64), nullable=False),
        sa.Column("species", sa.String(length=8), nullable=False),
        sa.Column("breed", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("source_ref", sa.Text(), nullable=True),
        sa.Column("licence", sa.String(length=64), nullable=True),
        sa.Column("attribution", sa.Text(), nullable=True),
        sa.Column("embedding", pgvector.sqlalchemy.vector.VECTOR(dim=384), nullable=False),
        sa.Column("embed_model", sa.String(length=128), nullable=False),
        sa.CheckConstraint("species IN ('cat', 'dog')", name=op.f("ck_train_image_species")),
        sa.PrimaryKeyConstraint("image_id", name=op.f("pk_train_image")),
    )
    op.create_index(
        "ix_train_image_species_breed", "train_image", ["species", "breed"], unique=False
    )
    op.add_column("detection", sa.Column("group_idx", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("detection", "group_idx")
    op.drop_index("ix_train_image_species_breed", table_name="train_image")
    op.drop_table("train_image")
