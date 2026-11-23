"""VLM screener tags (Step 13).

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-23 14:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "vlm_tag",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("image_id", sa.String(length=64), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("model_tag", sa.String(length=64), nullable=False),
        sa.Column("model_digest", sa.String(length=80), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=True),
        sa.Column("whole_animal", sa.Boolean(), nullable=True),
        sa.Column("kitten", sa.Boolean(), nullable=True),
        sa.Column("blurry", sa.Boolean(), nullable=True),
        sa.Column("several_animals", sa.Boolean(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("seconds", sa.Float(), nullable=False),
        sa.Column("raw", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('photo', 'drawing', 'toy', 'other')", name=op.f("ck_vlm_tag_kind")
        ),
        sa.CheckConstraint("status IN ('ok', 'parse_failed')", name=op.f("ck_vlm_tag_status")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_vlm_tag")),
        sa.UniqueConstraint(
            "image_id", "prompt_version", "model_tag", name=op.f("uq_vlm_tag_image_id")
        ),
    )
    op.create_index("ix_vlm_tag_image", "vlm_tag", ["image_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_vlm_tag_image", table_name="vlm_tag")
    op.drop_table("vlm_tag")
