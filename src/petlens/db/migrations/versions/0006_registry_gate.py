"""Promotion gate result and promotion time on the model registry (Step 11).

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-23 13:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "model_registry",
        sa.Column("gate", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "model_registry", sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("model_registry", "promoted_at")
    op.drop_column("model_registry", "gate")
