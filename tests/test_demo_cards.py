"""Step 17: the demo database URL and seed images, and the generated cards."""

from __future__ import annotations

import io
import os

import pytest
from PIL import Image

from petlens import cards, demo
from petlens.eval.fixtures import synthetic_rows

DATABASE_URL = os.environ.get("DATABASE_URL")


def test_demo_uses_its_own_database() -> None:
    url = demo.demo_url("postgresql+psycopg://petlens:secret@localhost:5433/petlens")
    assert url == "postgresql+psycopg://petlens:secret@localhost:5433/petlens_demo"


def test_demo_crops_are_small_webp_images() -> None:
    data = demo.crop_image(0.1, 2)
    assert len(data) < 40_000
    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "WEBP" and im.size == (192, 192)
    assert demo.crop_image(0.1, 1) != data


def test_cards_describe_the_data_and_the_run() -> None:
    rows, _ = synthetic_rows()
    rows[0].source, rows[0].author, rows[0].source_ref = "commons", "Jane", "https://x/1"
    rows[0].label_origin = "commons_category"
    text = cards.dataset_card("v1", rows)
    assert f"Photos: **{len(rows)}**" in text and "| commons |" in text
    attribution = cards.attribution("v1", rows)
    assert "- Jane, x: <https://x/1>" in attribution and "Parkhi" in attribution
    assert "No trained run" in cards.model_card(None, None)
    info = {
        "run_id": "r1",
        "dataset_version": "v1",
        "embedding_model": "vit",
        "metrics": {
            "cat": {
                "tests": {"test_domestic": {"calibrated": {"top1": 0.9}}},
                "policy": {"test_domestic": {"false_breed_claim_rate": 0.05}},
            }
        },
    }
    card = cards.model_card(info, {"passed": True, "numbers": {"golden set": {"diff": 0.02}}})
    assert "| cat | Ordinary cats (no breed) | 90.0% | 5.0% | n/a |" in card
    assert "gate passed" in card


@pytest.mark.db
@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_demo_seed_fills_a_separate_database(tmp_path: object) -> None:
    from pathlib import Path

    from sqlalchemy import create_engine, func, select

    from petlens.db import models as m

    assert DATABASE_URL is not None
    url = demo.ensure_database(DATABASE_URL)
    from petlens.eval import fixtures

    fixtures.main(url)
    assert demo.seed_pets(url, Path(str(tmp_path))) == 10
    engine = create_engine(url)
    with engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(m.Pet)) == 4
    engine.dispose()
    assert len(list(Path(str(tmp_path)).rglob("*.webp"))) == 10
