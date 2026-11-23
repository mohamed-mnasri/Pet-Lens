"""Run dbt against the app's database: ``python -m petlens.eval.dbt_run build``.

Reads DATABASE_URL (from .env too) and passes it to dbt's profile as PETLENS_DB_*,
so ``make dbt`` works whatever port Postgres is on.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy.engine import make_url

from petlens.config import get_settings

DBT_DIR = Path(__file__).resolve().parents[3] / "dbt"


def env_for(database_url: str) -> dict[str, str]:
    url = make_url(database_url)
    return {
        "PETLENS_DB_HOST": url.host or "localhost",
        "PETLENS_DB_PORT": str(url.port or 5432),
        "PETLENS_DB_USER": url.username or "petlens",
        "PETLENS_DB_PASSWORD": url.password or "",
        "PETLENS_DB_NAME": url.database or "petlens",
        "DBT_PROFILES_DIR": str(DBT_DIR),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv) or ["build"]
    env = os.environ | env_for(get_settings().database_url)
    return subprocess.call(["dbt", *args], cwd=DBT_DIR, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
