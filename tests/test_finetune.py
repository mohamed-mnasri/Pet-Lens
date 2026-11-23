"""Step 12: fine-tuning plumbing and the CI regression gate on a tiny fixture model."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from petlens.data.imaging import store_image
from petlens.eval.fixtures import synthetic_rows
from petlens.models import train
from test_models import BREEDS, _png, _row

REGRESSION = Path(__file__).parent / "data" / "regression_metrics.json"
TOLERANCE = 0.02


def _fixture_metrics() -> dict[str, float]:
    rows, emb = synthetic_rows(seed=1, noise=3.5)  # hard enough that metrics can move
    _, metrics = train.train(rows, emb, BREEDS, seed=0)
    out: dict[str, float] = {}
    for species in ("cat", "dog"):
        m = metrics[species]
        assert isinstance(m, dict)
        for test, t in m["tests"].items():
            for key in ("top1", "top3", "macro_f1", "ece"):
                out[f"{species}/{test}/{key}"] = round(float(t["calibrated"][key]), 4)
        pol = m["policy"]
        for test, e in pol.items():
            if isinstance(e, dict) and "coverage" in e:
                out[f"{species}/{test}/coverage"] = round(float(e["coverage"]), 4)
    return out


def test_tiny_fixture_model_reproduces_its_stored_metrics() -> None:
    """The regression gate: training on the fixed synthetic data must give the stored
    numbers. After an intended change, refresh them with ``make regression-update``."""
    got = _fixture_metrics()
    if os.environ.get("PETLENS_UPDATE_REGRESSION") == "1":
        REGRESSION.parent.mkdir(exist_ok=True)
        REGRESSION.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stored = json.loads(REGRESSION.read_text(encoding="utf-8"))
    assert set(got) == set(stored), "metrics added or removed: run make regression-update"
    off = {k: (stored[k], got[k]) for k in stored if abs(stored[k] - got[k]) > TOLERANCE}
    assert not off, f"metrics moved more than {TOLERANCE}: {off}"


def test_finetune_tunes_only_the_last_blocks(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    import torch
    from timm.models.vision_transformer import VisionTransformer

    from petlens.models import embed, finetune

    images = tmp_path / "processed" / "images"
    rows = []
    for i, (breed, split) in enumerate(
        [(b, s) for b in ("ragdoll", "birman") for s in ["train"] * 3 + ["val"]]
    ):
        stored = store_image(_png(i), images)
        rows.append(replace(_row(stored.image_id, breed, split, "oxford", "cat"), bbox=None))
    torch.manual_seed(0)
    tiny = VisionTransformer(
        img_size=224, patch_size=32, embed_dim=32, depth=3, num_heads=2, num_classes=0
    )
    before = {k: v.clone() for k, v in tiny.state_dict().items()}
    cfg = finetune.FinetuneConfig(unfreeze_blocks=1, epochs=2, batch_size=4, num_workers=0)
    out = tmp_path / "ft"
    info = finetune.finetune(
        rows, BREEDS, tmp_path, out, cfg, device="cpu", backbone=tiny, golden=frozenset()
    )
    assert info["train_images"] == 6 and info["val_images"] == 2
    assert len(info["history"]) == 2 and 0 <= info["best_val_bal_acc"] <= 1
    saved = torch.load(out / "backbone.pt", weights_only=True)
    assert torch.equal(saved["blocks.0.attn.qkv.weight"], before["blocks.0.attn.qkv.weight"])
    assert not torch.equal(saved["blocks.2.attn.qkv.weight"], before["blocks.2.attn.qkv.weight"])
    assert json.loads((out / "config.json").read_text())["base"] == embed.DEFAULT_MODEL

    name = finetune.model_name(cfg, "2026-11-23-abc")
    assert name == "ft-vits14-last1-2026-11-23-abc"
    assert embed.finetuned_dir(name, tmp_path) == tmp_path / "finetune" / name
    assert embed.finetuned_dir(embed.DEFAULT_MODEL, tmp_path) is None


def test_golden_and_unscreened_photos_stay_out_of_tuning() -> None:
    pytest.importorskip("torch")
    from petlens.models import finetune

    classes = finetune.joint_classes(BREEDS)
    rows = [
        _row("a", "ragdoll", "train", "oxford", "cat"),
        _row("gold", "ragdoll", "train", "oxford", "cat"),
        _row("web", "ragdoll", "train", "commons", "cat"),  # not admitted by screening
        _row("t", "ragdoll", "test_in_domain", "oxford", "cat"),
    ]
    items = finetune.split_items(rows, classes, frozenset(), frozenset({"gold"}), "train")
    assert [i[0] for i in items] == ["a"]
    assert classes[items[0][2]] == "cat:ragdoll"
    assert finetune.balanced_accuracy(np.array([0, 0, 1]), np.array([0, 1, 1])) == 0.75
