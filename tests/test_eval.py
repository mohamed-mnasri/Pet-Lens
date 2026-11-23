"""Evaluation module: metrics per test set, intervals, and the metric tables."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from petlens.eval import compute
from petlens.models import policy, train
from test_models import BREEDS, _row, _synthetic

DATABASE_URL = os.environ.get("DATABASE_URL")


def test_wilson_interval() -> None:
    lo, hi = compute.wilson(8, 10)
    assert lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)
    assert compute.wilson(0, 0) == (0.0, 1.0)
    assert compute.wilson(10, 10)[1] == pytest.approx(1.0)


def _fixture() -> compute.EvalResult:
    rows, emb, centres = _synthetic()
    rng = np.random.default_rng(3)
    for k in range(6):  # domestic test cats and two unknown breeds
        rows.append(_row(f"dom{k}", "domestic_cat", "test_domestic", "commons", "cat"))
        emb[f"dom{k}"] = (centres["domestic_cat"] + rng.normal(size=16)).astype(np.float32)
        rows.append(_row(f"unk{k}", "korat", "test_unknown", "commons", "cat"))
        emb[f"unk{k}"] = (rng.normal(size=16) * 4).astype(np.float32)
    heads, metrics = train.train(rows, emb, BREEDS)
    cfg = policy.PolicyConfig.model_validate(metrics["policy_config"])
    return compute.evaluate(rows, emb, heads, BREEDS, cfg)


def test_metrics_per_test_set() -> None:
    result = _fixture()
    by = {(m["test_set"], m["metric"], m["param"]): m for m in result.metrics}
    top1 = by[("test_in_domain", "top1", None)]
    # Up to 4 per class; near-copies of training photos are left out first.
    assert (
        22 < top1["n"] <= 22 * 4
        and top1["value"] > 0.8
        and top1["lo"] < top1["value"] <= top1["hi"] + 1e-9
    )
    assert ("test_in_domain", "risk_at_coverage", 0.5) in by
    assert 0 <= by[("test_domestic", "false_breed_claim_rate", None)]["value"] <= 1
    assert ("test_unknown", "unknown_rejection_rate", None) in by
    assert ("test_unknown", "top1", None) not in by  # no right answer for unknown breeds
    recalls = [c for c in result.class_metrics if c["test_set"] == "test_in_domain"]
    assert recalls and all(c["recall_lo"] <= c["recall"] <= c["recall_hi"] + 1e-9 for c in recalls)
    unknown = [i for i in result.images if i["test_set"] == "test_unknown"]
    assert len(unknown) == 6 and not any(i["known_breed"] or i["top1_correct"] for i in unknown)
    assert {i["outcome"] for i in result.images} <= {"accept", "no_clear_breed", "review"}


def test_composition_counts() -> None:
    rows, _, _ = _synthetic()
    comp = compute.composition(rows)
    assert sum(c["n"] for c in comp) == len(rows)
    assert {"source", "species", "breed", "split", "label_origin", "n"} == set(comp[0])


@pytest.mark.db
@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_results_are_stored(tmp_path: Path) -> None:
    from sqlalchemy import create_engine, func, select, text
    from sqlalchemy.orm import Session

    from petlens.db import migrate
    from petlens.db import models as m
    from petlens.eval import store

    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    migrate.upgrade(DATABASE_URL)
    result = _fixture()
    rows, _, _ = _synthetic()
    with Session(engine) as s:
        store.save(s, result, "run-1", "v1", "fake", compute.composition(rows))
        store.save(s, result, "run-1", "v1", "fake", compute.composition(rows))  # rerun
        s.commit()
        assert s.scalar(select(func.count()).select_from(m.EvalRun)) == 2
        assert s.scalar(select(func.count()).select_from(m.EvalImage)) == 2 * len(result.images)
        total = s.scalar(select(func.sum(m.DatasetCount.n)))
        assert total == len(rows)  # replaced per version, not doubled
    engine.dispose()
