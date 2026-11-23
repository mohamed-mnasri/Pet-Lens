"""Swipe review: cards, answers, reviewer checks, the API, and export into the dataset.

The database tests need Postgres (CI has it) and are skipped without DATABASE_URL.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from petlens import contracts as c
from petlens.api.app import create_app
from petlens.config import Settings
from petlens.data.catalog import load_breeds
from petlens.data.dataset import apply_reviews
from petlens.data.manifest import ManifestRow, read_manifest
from petlens.db import migrate
from petlens.db import models as m
from petlens.jobs import queue
from petlens.jobs.worker import Worker
from petlens.models.screening import ScreenResult
from petlens.models.train import in_training_pool
from petlens.review import answers, cards
from petlens.review.export import export
from test_api import _analyzer, _cat_photo

DATABASE_URL = os.environ.get("DATABASE_URL")


def _row(image_id: str, breed: str, source: str = "commons", **kw: object) -> ManifestRow:
    defaults: dict[str, object] = {
        "source_ref": image_id,
        "licence": "CC-BY-4.0",
        "attribution": "x",
        "species": "cat",
        "label_origin": "commons_category" if source == "commons" else source,
        "width": 500,
        "height": 400,
        "bbox": [0.1, 0.1, 0.9, 0.9],
        "split": "train",
    }
    return ManifestRow(image_id=image_id, source=source, breed=breed, **(defaults | kw))  # type: ignore[arg-type]


def _screen(image_id: str, label: str, predicted: str, decision: str) -> ScreenResult:
    return ScreenResult(image_id, "train", label, predicted, 0.3, 0.6, decision)


ROWS = [
    _row("a" * 64, "bengal"),
    _row("b" * 64, "ragdoll"),
    _row("c" * 64, "siamese"),
    _row("d" * 64, "persian"),
]
RESULTS = [
    _screen("a" * 64, "bengal", "domestic_cat", "review"),  # easy question first
    _screen("b" * 64, "ragdoll", "birman", "review"),
    _screen("c" * 64, "siamese", "siamese", "admit_and_review"),  # 5% sample
    _screen("d" * 64, "persian", "persian", "admit"),  # admitted, no card
]


# --- no database ------------------------------------------------------------------


def test_commons_cards_follow_screening() -> None:
    made = {cd["image_id"][0]: cd for cd in cards.commons_cards(ROWS, RESULTS)}
    assert set(made) == {"a", "b", "c"}
    assert made["a"]["priority"] < made["b"]["priority"] < made["c"]["priority"]
    assert made["b"]["suggested"] == "ragdoll" and made["b"]["options"] == ["birman"]
    assert made["a"]["x0"] == pytest.approx(0.1)


def test_check_cards_show_the_truth_only_sometimes() -> None:
    oxford = [
        _row(f"{i:064x}", "bengal", source="oxford", split="test_in_domain") for i in range(40)
    ]
    made = cards.check_cards(oxford, load_breeds(), n=40)
    assert len(made) == 40 and all(cd["truth"] == "bengal" for cd in made)
    shown_truth = sum(cd["suggested"] == "bengal" for cd in made)
    assert 10 < shown_truth < 30


def test_reviews_change_labels_and_training_pool() -> None:
    rows = [
        _row("a" * 64, "bengal"),
        _row("b" * 64, "ragdoll"),
        _row("c" * 64, "siamese"),
        _row("e" * 64, "maine_coon"),
    ]
    reviews = {
        "a" * 64: {"action": "fix", "label": "domestic_cat"},
        "b" * 64: {"action": "reject", "label": "not_pet"},
        "c" * 64: {"action": "unsure", "label": "siamese"},
        "e" * 64: {"action": "accept", "label": "maine_coon"},
    }
    out = {r.image_id[0]: r for r in apply_reviews(rows, reviews)}
    assert (out["a"].breed, out["a"].label_origin) == ("domestic_cat", "me")
    assert "review_rejected" in out["b"].flags
    assert "review_unsure" in out["c"].flags
    assert out["e"].label_origin == "me"
    admitted: frozenset[str] = frozenset({"c" * 64})
    assert in_training_pool(out["e"], frozenset())  # reviewed, so in
    assert not in_training_pool(out["c"], admitted)  # not sure beats screening
    audit = _row("f" * 64, "bengal", source="review", label_origin="me", flags=["audit_only"])
    assert not in_training_pool(audit, frozenset())


# --- with the database --------------------------------------------------------------


@pytest.fixture
def sessions() -> Iterator[sessionmaker[Session]]:
    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    migrate.upgrade(DATABASE_URL)
    yield sessionmaker(engine, expire_on_commit=False)
    engine.dispose()


def _load(s: Session) -> dict[str, uuid.UUID]:
    cards.insert_cards(s, cards.commons_cards(ROWS, RESULTS))
    check = cards.check_cards(
        [_row("9" * 64, "bengal", source="oxford", split="test_in_domain")], load_breeds(), 1
    )
    check[0]["suggested"] = "bengal"  # make the check card's right answer "correct"
    cards.insert_cards(s, check)
    assert cards.insert_cards(s, cards.commons_cards(ROWS, RESULTS)) == 0  # idempotent
    s.commit()
    return {cd.image_id[0]: cd.id for cd in s.scalars(select(m.ReviewCard)) if cd.image_id}


@pytest.mark.db
@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_answers_become_label_events(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        ids = _load(s)
        first = answers.next_card(s)
        assert first is not None and first.id == ids["a"]  # the easy domestic question
        assert first.question == "Is this a Bengal?"
        assert "domestic_cat" in first.options and first.options[-1] == "not_pet"
        with pytest.raises(answers.ReviewError, match="Pick an answer"):
            answers.answer(s, ids["a"], c.ReviewIn(answer="wrong"))
        e = answers.answer(s, ids["a"], c.ReviewIn(answer="wrong", label="domestic_cat"))
        assert (e.action, e.proposed_label, e.origin) == ("fix", "domestic_cat", "me")
        with pytest.raises(answers.ReviewError, match="already answered"):
            answers.answer(s, ids["a"], c.ReviewIn(answer="correct"))
        assert (
            answers.answer(s, ids["b"], c.ReviewIn(answer="wrong", label="not_pet")).action
            == "reject"
        )
        assert answers.answer(s, ids["c"], c.ReviewIn(answer="unsure")).action == "unsure"
        assert (
            answers.answer(s, ids["9"], c.ReviewIn(answer="correct", review_ms=900)).action
            == "accept"
        )
        s.commit()
        st = answers.stats(s)
        assert st.pending == {} and st.answered == 4
        assert (st.checks_answered, st.checks_correct, st.reviewer_accuracy) == (1, 1, 1.0)
        assert answers.next_card(s) is None


def test_blind_audit_and_export(tmp_path: Path) -> None:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    engine.dispose()
    migrate.upgrade(DATABASE_URL)
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "one.jpg").write_bytes(_cat_photo())
    (folder / "two.jpg").write_bytes(_cat_photo((320, 240)))
    settings = Settings(database_url=DATABASE_URL, data_dir=tmp_path / "data")
    analyzer = _analyzer()
    worker = Worker(settings, analyzer=lambda: analyzer)
    with worker.sessions() as s:
        queue.enqueue(s, "review_batch", {"folder": str(folder)})
        s.commit()
    worker.run(max_jobs=1)
    with worker.sessions() as s:
        made = s.scalars(select(m.ReviewCard).order_by(m.ReviewCard.ref)).all()
        assert {cd.queue for cd in made} <= {"batch", "audit"} and len(made) == 2
        # Make one of them a blind audit card to test that path.
        blind = made[0]
        blind.queue, blind.suggested = "audit", None
        s.commit()
        view = answers.next_card(s, "audit")
        assert view is not None and view.suggested is None and view.question.startswith("What")
        with pytest.raises(answers.ReviewError):
            answers.answer(s, blind.id, c.ReviewIn(answer="correct"))
        e = answers.answer(s, blind.id, c.ReviewIn(answer="wrong", label="ragdoll"))
        assert e.action == "accept"  # matched the hidden model answer
        other = made[1]
        answers.answer(s, other.id, c.ReviewIn(answer="wrong", label="bengal"))
        cards.insert_cards(s, cards.commons_cards(ROWS, RESULTS))
        commons_id = s.scalars(
            select(m.ReviewCard.id).where(m.ReviewCard.ref == f"commons:{'a' * 64}")
        ).one()
        answers.answer(s, commons_id, c.ReviewIn(answer="wrong", label="domestic_cat"))
        s.commit()
        summary = export(s, settings, repo_data_dir=tmp_path / "repo")
    assert "1 Commons reviews" in summary and "2 of my own photos" in summary
    labels = pq.read_table(tmp_path / "repo" / "reviews" / "labels.parquet").to_pylist()
    assert [(r["label"], r["action"]) for r in labels] == [("domestic_cat", "fix")]
    own = read_manifest(settings.processed_dir / "sources" / "review" / "manifest.parquet")
    by_breed = {r.breed: r for r in own}
    assert set(by_breed) == {"ragdoll", "bengal"}
    assert all(r.label_origin == "me" and r.source == "review" for r in own)
    assert "audit_only" in by_breed["ragdoll"].flags
    assert "audit_only" not in by_breed["bengal"].flags


def test_review_api(tmp_path: Path) -> None:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    engine.dispose()
    migrate.upgrade(DATABASE_URL)
    settings = Settings(database_url=DATABASE_URL, data_dir=tmp_path)
    analyzer = _analyzer()
    with TestClient(create_app(settings, lambda: analyzer)) as client:
        sessions = sessionmaker(create_engine(DATABASE_URL), expire_on_commit=False)
        with sessions() as s:
            _load(s)
        card = client.get("/review/next", params={"queue": "commons"}).json()
        assert card["queue"] == "commons" and card["suggested"] == "bengal"
        r = client.post(f"/review/{card['id']}", json={"answer": "correct", "review_ms": 1200})
        assert r.status_code == 200 and r.json()["action"] == "accept"
        again = client.post(f"/review/{card['id']}", json={"answer": "correct"})
        assert again.status_code == 409
        bad = client.post(f"/review/{uuid.uuid4()}", json={"answer": "correct"})
        assert bad.status_code == 404
        assert client.get("/review/stats").json()["answered"] == 1
        assert client.get("/review/next", params={"queue": "nope"}).status_code == 422


@pytest.mark.db
@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_weak_photos_leave_the_queue(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        ids = _load(s)
        answers.answer(s, ids["a"], c.ReviewIn(answer="unsure"))
        flagged = [_row(r.image_id, r.breed, flags=["grayscale"]) for r in ROWS]
        # "a" was answered, so it stays; "b" and "c" were pending and go.
        assert cards.drop_weak_cards(s, flagged) == 2
        s.commit()
        left = {cd.image_id[0] for cd in s.scalars(select(m.ReviewCard)) if cd.image_id}
        assert left == {"a", "9"}
