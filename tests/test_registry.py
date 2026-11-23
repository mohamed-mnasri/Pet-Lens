"""Step 11: golden set, paired bootstrap, promotion gate and the model registry."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from petlens.data.manifest import ManifestRow
from petlens.models import golden, policy, promotion, train
from petlens.models.runs import LoadedRun
from test_models import BREEDS, _row, _synthetic

DATABASE_URL = os.environ.get("DATABASE_URL")


def test_paired_bootstrap_sees_a_real_gain_and_ignores_noise() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(400) < 0.70
    better = a | (rng.random(400) < 0.5)  # fixes about half of a's mistakes
    diff, lo, hi = promotion.paired_bootstrap(a, better, n_boot=2000)
    assert diff > 0.1 and lo > 0 and lo <= diff <= hi
    same = a.copy()
    flip = rng.choice(400, 6, replace=False)
    same[flip] = ~same[flip]  # a few random changes either way: noise, not a gain
    _, lo, hi = promotion.paired_bootstrap(a, same, n_boot=2000)
    assert lo <= 0 <= hi
    assert promotion.paired_bootstrap(a[:0], a[:0]) == (0.0, 0.0, 0.0)


def _rows_with_tests() -> tuple[list[ManifestRow], dict[str, np.ndarray]]:
    rows, emb, centres = _synthetic()
    rng = np.random.default_rng(7)
    for breed, centre in centres.items():
        for k in range(6):
            iid = f"u-{breed}-{k}"
            split = "test_domestic" if breed == "domestic_cat" else "test_unseen_source"
            rows.append(replace(_row(iid, breed, split, "own", "cat"), label_origin="me"))
            emb[iid] = (centre + rng.normal(size=16)).astype(np.float32)
    return rows, emb


def test_golden_set_is_hand_checked_held_out_and_frozen(tmp_path: Path) -> None:
    rows, _ = _rows_with_tests()
    rows.append(
        replace(_row("weak", "ragdoll", "test_in_domain", "oxford", "cat"), flags=["blurry"])
    )
    rows.append(_row("web", "ragdoll", "test_unseen_source", "openverse", "cat"))  # unchecked
    picked = golden.select(rows, BREEDS, "v1", per_breed=3)
    by_id = {r.image_id: r for r in rows}
    assert picked and all(by_id[g.image_id].split != "train" for g in picked)
    assert {"weak", "web"}.isdisjoint(g.image_id for g in picked)
    assert max(sum(g.breed == b for g in picked) for b in {g.breed for g in picked}) == 3
    assert picked == golden.select(rows, BREEDS, "v1", per_breed=3)  # reproducible
    path = tmp_path / "golden.parquet"
    golden.write(picked, path)
    assert golden.read(path) == picked
    with pytest.raises(FileExistsError):
        golden.write(picked, path)  # frozen
    assert "golden set: " in golden.summary(picked)


def _run(name: str, rows: list[ManifestRow], emb: dict[str, np.ndarray]) -> LoadedRun:
    heads, metrics = train.train(rows, emb, BREEDS)
    info = {"run_id": name, "dataset_version": "v1", "embedding_model": "fake", "metrics": metrics}
    cfg = policy.PolicyConfig.model_validate(metrics["policy_config"])
    return LoadedRun(Path(name), info, heads, cfg, rows, frozenset())


def test_gate_promotes_a_better_model_and_keeps_a_worse_one() -> None:
    rows, emb = _rows_with_tests()
    # The old model learnt from training photos with Ragdoll, Birman, Persian and Siamese
    # mixed up, so it confuses them; the new one learnt from clean labels.
    swap = {"ragdoll": "birman", "birman": "ragdoll", "persian": "siamese", "siamese": "persian"}
    noisy = [
        replace(r, breed=swap.get(r.breed, r.breed)) if r.split == "train" else r for r in rows
    ]
    old, new = _run("old", noisy, emb), _run("new", rows, emb)
    gold = golden.select(rows, BREEDS, "v1", per_breed=4)
    cfg = promotion.GateConfig(n_boot=2000, min_breed_n=4)
    vecs = {"fake": emb}

    up = promotion.compare(old, new, gold, Path("."), BREEDS, cfg, vecs)
    checks = {c.name: c for c in up.checks}
    assert checks["golden set"].passed and checks["unseen-source test"].passed
    assert checks["no breed falls behind"].passed
    assert up.numbers["golden set"]["challenger_top1"] > up.numbers["golden set"]["champion_top1"]

    down = promotion.compare(new, old, gold, Path("."), BREEDS, cfg, vecs)
    assert not down.passed
    checks = {c.name: c for c in down.checks}
    assert not checks["golden set"].passed
    assert (
        not checks["no breed falls behind"].passed
        and "ragdoll" in checks["no breed falls behind"].detail
    )
    assert "KEEP THE CHAMPION" in down.render()

    first = promotion.compare(None, new, gold, Path("."), BREEDS, cfg, vecs)
    assert first.passed and first.champion is None


def test_gate_refuses_a_model_trained_on_golden_photos() -> None:
    rows, emb = _rows_with_tests()
    gold = golden.select(rows, BREEDS, "v1", per_breed=2)
    gold_ids = {g.image_id for g in gold}
    leaky = [replace(r, split="train") if r.image_id in gold_ids else r for r in rows]
    result = promotion.compare(
        None, _run("leaky", leaky, emb), gold, Path("."), BREEDS, vectors={"fake": emb}
    )
    assert not result.passed and "golden" in result.checks[0].name


@pytest.mark.db
@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_registry_keeps_one_champion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy import create_engine, func, select, text
    from sqlalchemy.orm import Session

    from petlens.db import migrate
    from petlens.db import models as m
    from petlens.models import registry

    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    migrate.upgrade(DATABASE_URL)
    rows, emb = _rows_with_tests()
    first, second = _run("run-a", rows, emb), _run("run-b", rows, emb)
    with Session(engine) as s:
        a = registry.register(s, first)
        registry.register(s, first)  # idempotent
        registry.record_gate(s, a, {"passed": True}, promote=True)
        b = registry.register(s, second)
        registry.record_gate(s, b, {"passed": False}, promote=True)  # failed: no change
        best = registry.champion(s)
        assert best is not None and best.run_id == "run-a"
        registry.record_gate(s, b, {"passed": True}, promote=True)
        best = registry.champion(s)
        assert best is not None and best.run_id == "run-b" and best.promoted_at is not None
        assert s.scalar(select(func.count()).select_from(m.ModelRegistry)) == 2
        assert best.config["embedding_model"] == "fake" and "cat" in best.metrics
        s.commit()
    engine.dispose()

    models = tmp_path / "models"
    for name in ("run-a", "run-b", "run-c"):
        (models / "runs" / name).mkdir(parents=True)
        (models / "runs" / name / "heads.joblib").write_bytes(b"")
    assert registry.serving_run(models).name == "run-b"  # the champion, not the latest
