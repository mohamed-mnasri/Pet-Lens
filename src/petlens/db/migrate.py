"""Run Alembic migrations without an alembic.ini (paths come from the package)."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config

from petlens.config import get_settings

MIGRATIONS = Path(__file__).parent / "migrations"


def alembic_config(url: str | None = None) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    # ConfigParser interpolation: a literal % in a password must be doubled.
    cfg.set_main_option("sqlalchemy.url", (url or get_settings().database_url).replace("%", "%%"))
    return cfg


def upgrade(url: str | None = None, revision: str = "head") -> None:
    command.upgrade(alembic_config(url), revision)


def downgrade(url: str | None = None, revision: str = "base") -> None:
    command.downgrade(alembic_config(url), revision)


def revision(message: str, url: str | None = None) -> None:
    """Autogenerate a new migration by comparing the models with the database."""
    command.revision(alembic_config(url), message=message, autogenerate=True)
