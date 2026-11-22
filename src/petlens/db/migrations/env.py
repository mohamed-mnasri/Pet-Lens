"""Alembic environment: online migrations against DATABASE_URL."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from petlens.db.models import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise SystemExit("offline migrations are not supported; set DATABASE_URL")
run_migrations_online()
