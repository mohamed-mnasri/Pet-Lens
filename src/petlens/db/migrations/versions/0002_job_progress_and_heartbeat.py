"""Job progress, result and heartbeat (Step 7).

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-22 23:36:51.400254
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "job",
        sa.Column(
            "result",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "job", sa.Column("progress_done", sa.Integer(), server_default="0", nullable=False)
    )
    op.add_column("job", sa.Column("progress_total", sa.Integer(), nullable=True))
    op.add_column("job", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("job", "heartbeat_at")
    op.drop_column("job", "progress_total")
    op.drop_column("job", "progress_done")
    op.drop_column("job", "result")
