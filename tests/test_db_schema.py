"""Schema: contracts mirror the tables, and the migrations match the models.

The migration tests need Postgres with pgvector (CI has it). They are skipped unless
DATABASE_URL is set, and they drop and recreate the public schema, so point
DATABASE_URL at a throwaway database.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from petlens import contracts as c
from petlens.db import migrate
from petlens.db import models as m

DATABASE_URL = os.environ.get("DATABASE_URL")
needs_db = pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")

PAIRS = [
    (m.Upload, c.Upload),
    (m.Frame, c.Frame),
    (m.Detection, c.Detection),
    (m.Prediction, c.Prediction),
    (m.LabelEvent, c.LabelEvent),
    (m.Job, c.Job),
    (m.DatasetVersion, c.DatasetVersion),
    (m.ModelRegistry, c.ModelRegistry),
    (m.DeadLetter, c.DeadLetter),
]
NOT_IN_CONTRACT = {("detection", "embedding")}  # 384 floats stay out of API payloads


@pytest.mark.parametrize(("model", "contract"), PAIRS, ids=lambda x: getattr(x, "__name__", ""))
def test_contract_mirrors_table(model: type[m.Base], contract: type[c.Contract]) -> None:
    table = model.metadata.tables[model.__tablename__]
    columns = {col.name for col in table.columns} - {
        col for t, col in NOT_IN_CONTRACT if t == table.name
    }
    assert columns == set(contract.model_fields)
    for col in table.columns:
        if col.name in contract.model_fields:
            field = contract.model_fields[col.name]
            optional = not field.is_required() and field.default is None
            assert col.nullable == optional, f"{table.name}.{col.name} nullability differs"


def test_check_constraints_use_the_contract_literals() -> None:
    upload = m.Base.metadata.tables["upload"]
    sql = {str(ck.sqltext) for ck in upload.constraints if hasattr(ck, "sqltext")}
    assert "status IN ('received', 'stored', 'processing', 'done', 'rejected', 'failed')" in sql


@pytest.fixture
def engine() -> Iterator[Engine]:
    assert DATABASE_URL is not None
    eng = create_engine(DATABASE_URL)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    migrate.upgrade(DATABASE_URL)
    yield eng
    eng.dispose()


@needs_db
@pytest.mark.db
def test_migrations_match_the_models(engine: Engine) -> None:
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), m.Base.metadata)
    assert diff == []


@needs_db
@pytest.mark.db
def test_upgrade_downgrade_upgrade(engine: Engine) -> None:
    assert DATABASE_URL is not None
    migrate.downgrade(DATABASE_URL, "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    migrate.upgrade(DATABASE_URL)
    assert {"upload", "frame", "detection", "job", "dead_letter"} <= set(
        inspect(engine).get_table_names()
    )


@needs_db
@pytest.mark.db
def test_constraints_and_append_only(engine: Engine) -> None:
    with engine.begin() as conn:
        with pytest.raises(IntegrityError), conn.begin_nested():
            conn.execute(
                text(
                    "INSERT INTO upload (kind, original_name, mime, size_bytes, sha256) "
                    "VALUES ('podcast', 'a.mp3', 'audio/mpeg', 1, repeat('a', 64))"
                )
            )
        conn.execute(
            text(
                "INSERT INTO label_event (image_id, species, proposed_label, origin, action) "
                "VALUES (repeat('b', 64), 'cat', 'ragdoll', 'me', 'accept')"
            )
        )
        with pytest.raises(DBAPIError, match="append-only"), conn.begin_nested():
            conn.execute(text("UPDATE label_event SET proposed_label = 'bengal'"))
        with pytest.raises(DBAPIError, match="append-only"), conn.begin_nested():
            conn.execute(text("DELETE FROM label_event"))
        row = conn.execute(text("SELECT * FROM label_event")).mappings().one()
    event = c.LabelEvent.model_validate(dict(row))
    assert event.proposed_label == "ragdoll"
