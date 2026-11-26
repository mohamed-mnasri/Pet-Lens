"""Steps 13 and 14: the VLM screener and the breed-naming experiment (fake VLM, no GPU)."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from petlens.data import vlm_flags
from petlens.data.imaging import store_image
from petlens.models import train
from petlens.vlm import breeds, measure, screen
from test_models import BREEDS, _png, _row

DATABASE_URL = os.environ.get("DATABASE_URL")
GOOD = json.dumps(
    {
        "kind": "photo",
        "whole_animal": True,
        "kitten": False,
        "blurry": False,
        "several_animals": False,
    }
)


class FakeVlm:
    model = "fake-vl:1b"

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls: list[dict[str, Any]] = []

    def digest(self) -> str:
        return "sha256:abc"

    def ask(self, prompt: str, image_jpeg: bytes, schema: dict[str, Any]) -> str:
        self.calls.append({"prompt": prompt, "schema": schema, "bytes": len(image_jpeg)})
        return self.answers[min(len(self.calls), len(self.answers)) - 1]


def test_answers_are_validated_with_one_retry() -> None:
    assert screen.parse("```json\n" + GOOD + "\n```") is not None
    assert screen.parse('{"kind": "painting"}') is None
    retried = screen.screen_one(FakeVlm(["not json", GOOD]), b"x")
    assert retried.tags is not None and retried.attempts == 2
    failed = screen.screen_one(FakeVlm(["nope"]), b"x")
    assert failed.tags is None and failed.attempts == 2  # then parse_failed


class TimeoutVlm(FakeVlm):
    def ask(self, prompt: str, image_jpeg: bytes, schema: dict[str, Any]) -> str:
        self.calls.append({})
        if len(self.calls) == 1:
            raise httpx.ReadTimeout("timed out")
        return self.answers[0]


def test_a_timeout_counts_as_a_failed_answer() -> None:
    retried = screen.screen_one(TimeoutVlm([GOOD]), b"x")
    assert retried.tags is not None and retried.attempts == 2
    stuck = screen.screen_one(TimeoutVlm(["<never parsed>"]), b"x")
    assert stuck.tags is None and stuck.attempts == 2 and stuck.raw == "<never parsed>"


def test_ollama_request_uses_the_schema_and_the_image() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3-vl:4b", "digest": "d1"}]})
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": GOOD}})

    http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    vlm = screen.OllamaVlm(http=http)
    assert vlm.digest() == "d1"
    assert screen.parse(vlm.ask(screen.PROMPT, b"jpeg", screen.SCHEMA)) is not None
    body = seen[0]
    assert body["format"]["properties"]["kind"]["enum"] == ["photo", "drawing", "toy", "other"]
    assert body["messages"][0]["images"] and body["options"]["temperature"] == 0
    with pytest.raises(RuntimeError, match="vlm-pull"):
        screen.OllamaVlm("other:1b", http=http).digest()


def test_cohen_kappa() -> None:
    assert measure.cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    k = measure.cohen_kappa([True, True, False, False], [True, False, True, False])
    assert k == pytest.approx(0.0)
    assert measure.cohen_kappa([True, True], [True, True]) is None  # no variation
    assert measure.cohen_kappa([], []) is None


def test_tags_become_flags_and_keep_weak_web_photos_out_of_training() -> None:
    drawing = {
        "kind": "drawing",
        "whole_animal": True,
        "kitten": True,
        "blurry": False,
        "several_animals": False,
    }
    assert vlm_flags.flags_from(drawing) == {"vlm_not_photo", "vlm_kitten"}
    web = _row("w", "ragdoll", "train", "commons", "cat")
    mine = replace(_row("m", "ragdoll", "train", "commons", "cat"), label_origin="me")
    oxford = _row("o", "ragdoll", "train", "oxford", "cat")
    tagged = vlm_flags.apply_tags([web, mine, oxford], {i: drawing for i in "wmo"})
    assert all("vlm_not_photo" in r.flags for r in tagged)
    blocked = {r.image_id for r in tagged if vlm_flags.blocked(r)}
    assert blocked == {"w"}  # my reviewed photo and Oxford are never blocked
    admitted = frozenset({"w"})
    assert not train.in_training_pool(tagged[0], admitted)
    assert train.in_training_pool(tagged[1], admitted) and train.in_training_pool(
        tagged[2], admitted
    )
    # Tags are replaced, not piled up, when the file changes.
    again = vlm_flags.apply_tags(tagged, {})
    assert not any(set(r.flags) & vlm_flags.VLM_FLAGS for r in again)


def test_breed_question_is_limited_to_the_known_breeds() -> None:
    options = breeds.choices(BREEDS, "cat")
    assert ("domestic_cat", "domestic cat, no clear breed") in options
    vlm = FakeVlm(['{"breed": "siberian_husky"}', '{"breed": "ragdoll"}'])
    assert breeds.ask(vlm, b"x", "cat", options) == "ragdoll"  # out-of-list answer retried
    assert "ragdoll" in vlm.calls[0]["schema"]["properties"]["breed"]["enum"]
    assert breeds.ask(FakeVlm(["?"]), b"x", "cat", options) is None
    text = breeds.summarise(
        [
            {
                "test_set": "golden",
                "truth": "ragdoll",
                "vlm": "ragdoll",
                "classifier": "birman",
                "seconds": 2.0,
            },
            {
                "test_set": "golden",
                "truth": "birman",
                "vlm": "ragdoll",
                "classifier": "birman",
                "seconds": 3.0,
            },
        ],
        "fake-vl:1b",
        "run-1",
    )
    assert "| golden | 2 | 50.0%" in text and "birman called ragdoll (1)" in text


@pytest.mark.db
@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_screen_run_stores_resumes_exports_and_reports(tmp_path: Path) -> None:
    from sqlalchemy import create_engine, func, select, text
    from sqlalchemy.orm import Session

    from petlens.db import migrate
    from petlens.db import models as m

    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    migrate.upgrade(DATABASE_URL)
    images = tmp_path / "processed" / "images"
    rows = []
    for i in range(4):
        stored = store_image(_png(i), images)
        rows.append(_row(stored.image_id, "ragdoll", "train", "commons", "cat"))
    rows.append(_row("o" * 64, "ragdoll", "train", "oxford", "cat"))  # not screened
    bad = json.dumps(
        {
            "kind": "toy",
            "whole_animal": False,
            "kitten": False,
            "blurry": True,
            "several_animals": False,
        }
    )
    with Session(engine) as s:
        s.add(
            m.LabelEvent(
                image_id=rows[3].image_id,
                species="cat",
                proposed_label="ragdoll",
                origin="me",
                action="reject",
                review_ms=1,
            )
        )
        s.commit()
        assert screen.candidates(rows, {rows[3].image_id}, set())[0] is rows[3]  # reviewed first
        counts = screen.run(s, rows, tmp_path, FakeVlm([bad, GOOD, "x", "x"]), limit=2)
        assert counts == {"ok": 2, "parse_failed": 0, "missing": 0}
        counts = screen.run(s, rows, tmp_path, FakeVlm(["x"]))  # resumes with the other two
        assert counts["parse_failed"] == 2
        assert s.scalar(select(func.count()).select_from(m.VlmTag)) == 4
        path = tmp_path / "tags.parquet"
        assert screen.export(s, "fake-vl:1b", path) == 2
        tags = vlm_flags.read_tags(path)
        assert vlm_flags.flags_from(tags[rows[3].image_id]) == {
            "vlm_not_photo",
            "vlm_partial",
            "vlm_blurry",
        }
        report = measure.report(s, rows, "fake-vl:1b")
        assert "parse_failed: 2" in report and "Cohen's kappa" in report
    engine.dispose()
