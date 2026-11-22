"""Checks that Postgres is reachable and the pgvector extension is installed.

Skipped unless DATABASE_URL is set, so `make test` works without Docker running.
Run `make db-up` and set DATABASE_URL (or use .env) to include it locally.
"""

import os

import pytest
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.db

DATABASE_URL = os.environ.get("DATABASE_URL")


@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_pgvector_extension_installed() -> None:
    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
    engine.dispose()
    assert version is not None, "pgvector extension is not installed"
