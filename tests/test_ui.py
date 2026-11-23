"""Upload page: pre-checks, wording, drawing, the API client, and a page smoke test."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from petlens.ui import view

LIMITS = {
    "photo_max_bytes": 15 * 2**20,
    "photo_max_pixels": 40_000_000,
    "video_max_bytes": 25 * 2**20,
    "video_max_seconds": 10.0,
}


def _jpeg(size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (100, 100, 100)).save(buf, "JPEG")
    return buf.getvalue()


def test_precheck_catches_problems_before_upload() -> None:
    assert view.precheck("a.jpg", _jpeg(), LIMITS) == ("photo", None)
    assert view.precheck("clip.mp4", b"\0\0\0\x18ftypisom", LIMITS) == ("video", None)
    kind, problem = view.precheck("a.gif", b"GIF89a", LIMITS)
    assert kind == "unknown" and problem is not None and "not a JPEG" in problem
    _, problem = view.precheck("big.jpg", b"\xff\xd8\xff" + b"\0" * (15 * 2**20), LIMITS)
    assert problem is not None and "15 MB" in problem
    huge = io.BytesIO()
    Image.new("RGB", (8000, 5001)).save(huge, "PNG")
    _, problem = view.precheck("huge.png", huge.getvalue(), LIMITS)
    assert problem is not None and "megapixels" in problem


def _pred(**kw: Any) -> dict[str, Any]:
    base = {
        "id": "p1",
        "outcome": "accept",
        "label": "maine_coon",
        "also": None,
        "confidence": 0.91,
        "species": "cat",
        "detection_id": "d1",
        "group_idx": None,
    }
    return base | kw


def test_wording() -> None:
    assert view.headline(_pred()) == "Looks like a Maine Coon, 91%"
    assert view.headline(_pred(also="norwegian_forest")) == (
        "Looks like a Maine Coon (or Norwegian Forest), 91%"
    )
    assert view.headline(_pred(outcome="no_clear_breed")) == "Domestic cat, no clear breed"
    assert view.headline(_pred(outcome="no_clear_breed", species="dog")) == "No clear breed"
    assert view.headline(_pred(outcome="review")) == "Not sure: sent to review"
    assert view.badge(_pred(outcome="review")) == ("Not sure", "gray")
    assert "10 s" in view.limits_text(LIMITS) and "40 MP" in view.limits_text(LIMITS)


def test_labels_and_boxes() -> None:
    run = {
        "predictions": [_pred(), _pred(id="p2", detection_id=None, group_idx=0, outcome="review")],
        "detections": [
            {"id": "d1", "species": "cat", "x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.6},
            {
                "id": "d2",
                "species": "cat",
                "group_idx": 0,
                "x0": 0.5,
                "y0": 0.2,
                "x1": 0.9,
                "y1": 0.9,
            },
        ],
    }
    labels = view.detection_labels(run)
    assert labels == {"d1": "Maine Coon", "d2": "#1 not sure"}
    png = view.draw_boxes(_jpeg((200, 100)), run["detections"], labels)
    with Image.open(io.BytesIO(png)) as im:
        assert im.size == (200, 100)
        assert im.getpixel((20, 50)) != (100, 100, 100)  # the box edge was drawn


def test_job_progress() -> None:
    assert view.job_progress({"progress_done": 3, "progress_total": 4, "state": "running"}) == 0.75
    assert view.job_progress({"progress_done": 0, "progress_total": None, "state": "done"}) == 1.0


DATABASE_URL = os.environ.get("DATABASE_URL")


@pytest.mark.db
@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_end_to_end_smoke_through_the_page_client(tmp_path: Path) -> None:
    """Photo in, decision and evidence out, through the same client the page uses, with
    tiny fixture models (a colour detector, colour embeddings and a logistic head)."""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine, text

    from petlens.api.app import create_app
    from petlens.config import Settings
    from petlens.db import migrate
    from petlens.ui.client import ApiClient, ApiError
    from test_api import _analyzer, _cat_photo, _seed_index

    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    engine.dispose()
    migrate.upgrade(DATABASE_URL)
    analyzer = _analyzer()
    app = create_app(Settings(database_url=DATABASE_URL, data_dir=tmp_path), lambda: analyzer)
    with TestClient(app) as http:
        _seed_index(tmp_path)
        api = ApiClient(http=http)
        assert api.health()
        limits = api.limits()
        kind, problem = view.precheck("cat.jpg", _cat_photo(), limits)
        assert (kind, problem) == ("photo", None)
        run = api.predict(kind, "cat.jpg", _cat_photo())
        pred = run["predictions"][0]
        assert view.headline(pred).startswith("Looks like a Ragdoll")
        assert len(run["evidence"][pred["id"]]) == 5
        frame = api.image(f"/images/frames/{run['frames'][0]['image_id']}")
        labels = view.detection_labels(run)
        assert view.draw_boxes(frame, run["detections"], labels)[:4] == b"\x89PNG"
        assert api.run(run["upload"]["id"])["upload"]["id"] == run["upload"]["id"]
        with pytest.raises(ApiError) as exc:
            api.predict("photo", "x.gif", b"GIF89a")
        assert exc.value.status == 415


class _FakeApi:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def health(self) -> bool:
        return True

    def limits(self) -> dict[str, float]:
        return LIMITS

    def review_stats(self) -> dict[str, Any]:
        return {
            "pending": {"commons": 3, "check": 1},
            "answered": 7,
            "checks_answered": 2,
            "checks_correct": 1,
            "reviewer_accuracy": 0.5,
        }

    def review_next(
        self, queue: str | None = None, species: str | None = None
    ) -> dict[str, Any] | None:
        return {
            "id": "card-1",
            "queue": "commons",
            "image_url": "/images/train/x",
            "box": None,
            "species": "cat",
            "suggested": "bengal",
            "question": "Is this a Bengal?",
            "options": ["domestic_cat", "not_pet"],
            "references": [],
            "pending": 4,
        }

    def image(self, url: str) -> bytes:
        return _jpeg()

    def breeds(self) -> list[dict[str, str]]:
        return [{"id": "ragdoll", "name": "Ragdoll", "species": "cat"}]

    def training_photo_counts(self) -> dict[str, int]:
        return {"birman": 2, "ragdoll": 5}

    def jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "kind": "review_batch",
                "state": "running",
                "progress_done": 2,
                "progress_total": 5,
                "error": None,
                "attempts": 1,
                "max_attempts": 3,
            }
        ]


def test_page_renders_limits_and_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from petlens.ui import client

    monkeypatch.setattr(client, "ApiClient", _FakeApi)
    page = Path(__file__).parents[1] / "src" / "petlens" / "ui" / "app.py"
    at = AppTest.from_file(str(page), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "🐾 PetLens"
    assert any("40 MP" in c.value for c in at.caption)
    assert any("face and the body" in i.value for i in at.info)
    assert any("review_batch" in (p.proto.text or "") for p in at.get("progress"))
    # The review tab: the question, my accuracy, and the three swipe buttons.
    assert any("Is this a Bengal?" in m.value for m in at.markdown)
    assert any("accuracy on check cards 50%" in c.value for c in at.caption)
    assert any("Ragdoll 5" in c.value and "7 photos" in c.value for c in at.caption)
    assert any(s.label.startswith("Breed of every photo") for s in at.selectbox)
    labels = [b.label for b in at.button]
    assert {"Wrong", "Not sure", "Correct"} <= set(labels)
    # After "Wrong": the card's choices plus a box to type the breed.
    next(b for b in at.button if b.label == "Wrong").click().run()
    assert not at.exception
    assert any(s.label == "Or type the breed" for s in at.selectbox)
    assert {"Not a cat or dog", "Skip"} <= {b.label for b in at.button}


def test_review_wording() -> None:
    assert view.option_label("not_pet") == "Not a cat or dog"
    assert view.option_label("domestic_cat") == "Domestic / no clear breed"
    text = view.stats_text({"pending": {"batch": 2, "check": 1}, "answered": 0})
    assert text.startswith("3 cards left") and "My batches 2" in text and "Check" not in text
