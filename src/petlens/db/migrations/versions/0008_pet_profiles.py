"""Pet profiles: pet, append-only identity_event, detection.pet_id (Step 15).

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-23 15:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APPEND_ONLY = """
CREATE FUNCTION identity_event_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'identity_event is append-only';
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER identity_event_append_only
    BEFORE UPDATE OR DELETE ON identity_event
    FOR EACH ROW EXECUTE FUNCTION identity_event_append_only();
"""


def upgrade() -> None:
    op.create_table(
        "pet",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("species", sa.String(length=8), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=16), server_default="candidate", nullable=False),
        sa.Column("merged_into", sa.UUID(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("species IN ('cat', 'dog')", name=op.f("ck_pet_species")),
        sa.CheckConstraint(
            "status IN ('candidate', 'named', 'merged')", name=op.f("ck_pet_status")
        ),
        sa.ForeignKeyConstraint(["merged_into"], ["pet.id"], name=op.f("fk_pet_merged_into_pet")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_pet")),
    )
    op.create_table(
        "identity_event",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("pet_id", sa.UUID(), nullable=False),
        sa.Column("detection_id", sa.UUID(), nullable=True),
        sa.Column("other_pet_id", sa.UUID(), nullable=True),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("origin", sa.String(length=8), nullable=False),
        sa.Column("value", sa.String(length=64), nullable=True),
        sa.Column("distance", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "action IN ('assign', 'new_pet', 'confirm', 'move', 'name', 'merge', 'split')",
            name=op.f("ck_identity_event_action"),
        ),
        sa.CheckConstraint("origin IN ('model', 'me')", name=op.f("ck_identity_event_origin")),
        sa.ForeignKeyConstraint(
            ["detection_id"],
            ["detection.id"],
            name=op.f("fk_identity_event_detection_id_detection"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["other_pet_id"], ["pet.id"], name=op.f("fk_identity_event_other_pet_id_pet")
        ),
        sa.ForeignKeyConstraint(["pet_id"], ["pet.id"], name=op.f("fk_identity_event_pet_id_pet")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_identity_event")),
    )
    op.create_index("ix_identity_event_detection", "identity_event", ["detection_id"])
    op.execute(APPEND_ONLY)
    op.add_column("detection", sa.Column("pet_id", sa.UUID(), nullable=True))
    op.create_foreign_key(
        op.f("fk_detection_pet_id_pet"), "detection", "pet", ["pet_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index(op.f("ix_detection_pet_id"), "detection", ["pet_id"])


def downgrade() -> None:
    op.drop_index(op.f("ix_detection_pet_id"), table_name="detection")
    op.drop_constraint(op.f("fk_detection_pet_id_pet"), "detection", type_="foreignkey")
    op.drop_column("detection", "pet_id")
    op.execute("DROP TRIGGER IF EXISTS identity_event_append_only ON identity_event")
    op.execute("DROP FUNCTION IF EXISTS identity_event_append_only()")
    op.drop_index("ix_identity_event_detection", table_name="identity_event")
    op.drop_table("identity_event")
    op.drop_table("pet")
