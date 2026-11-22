"""Baseline heads, calibration, metrics and embedding plumbing (no torch needed)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from petlens.data.catalog import load_breeds
from petlens.data.imaging import store_image
from petlens.data.manifest import ManifestRow
from petlens.models import embed, train
from petlens.models.calibration import fit_temperature
from petlens.models.metrics import expected_calibration_error, macro_f1, softmax, top_k_accuracy

BREEDS = load_breeds()


def test_metrics_on_known_values() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
    y = np.array([0, 1, 1])
    assert top_k_accuracy(probs, y, 1) == pytest.approx(1 / 3)
    assert top_k_accuracy(probs, y, 2) == pytest.approx(1.0)
    assert macro_f1(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), 2) == pytest.approx(
        np.mean([2 / 3, 0.8])
    )
    perfect = np.eye(3)
    assert expected_calibration_error(perfect, np.arange(3)) == pytest.approx(0.0)


def test_temperature_recovers_known_value() -> None:
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 5)) * 2
    p = softmax(true_logits)
    y = np.array([rng.choice(5, p=row) for row in p])
    # The model is overconfident by a factor of 3: logits = 3 * true logits.
    t = fit_temperature(true_logits * 3, y)
    assert t == pytest.approx(3.0, rel=0.1)


def _row(image_id: str, breed: str, split: str, source: str, species: str) -> ManifestRow:
    return ManifestRow(
        image_id=image_id,
        source=source,
        source_ref=image_id,
        licence="x",
        attribution="x",
        species=species,
        breed=breed,
        label_origin=source,
        width=1,
        height=1,
        split=split,
    )


def test_train_end_to_end_on_synthetic_embeddings() -> None:
    cats = BREEDS.training_classes("cat")
    dogs = BREEDS.training_classes("dog")
    rng = np.random.default_rng(1)
    centres = {c: rng.normal(size=16) * 4 for c in cats + dogs}
    rows, emb = [], {}
    splits = ["train"] * 12 + ["val"] * 3 + ["calibration"] * 3 + ["test_in_domain"] * 4
    k = 0
    for species, classes in (("cat", cats), ("dog", dogs)):
        for c in classes:
            for s in splits:
                source = "commons" if c == "domestic_cat" else "oxford"
                if c == "domestic_cat" and s == "test_in_domain":
                    s = "test_domestic"
                k += 1
                rows.append(_row(f"i{k}", c, s, source, species))
                emb[f"i{k}"] = (centres[c] + rng.normal(size=16)).astype(np.float32)
    # An unscreened Commons pedigree image in "train" must not be trained on.
    rows.append(_row("unscreened", "ragdoll", "train", "commons", "cat"))
    emb["unscreened"] = centres["persian"].astype(np.float32)

    heads, metrics = train.train(rows, emb, BREEDS)
    assert set(heads) == {"cat", "dog"}
    cat = metrics["cat"]
    assert isinstance(cat, dict)
    assert cat["split_sizes"]["train"] == 13 * 12
    assert cat["tests"]["test_in_domain"]["calibrated"]["top1"] > 0.9
    assert cat["domestic_test_called_a_breed"] < 0.5
    assert [p["fraction"] for p in cat["learning_curve"]] == [0.25, 0.5, 1.0]
    report = train.render_report("run", "v1", "fake", metrics)
    assert "| test_in_domain |" in report and "Learning curve" in report


def test_crop_for_model_uses_box_and_pads_square() -> None:
    img = Image.new("RGB", (400, 200), (0, 0, 255))
    img.paste((255, 0, 0), (100, 50, 300, 150))
    out = embed.crop_for_model(img, [0.25, 0.25, 0.75, 0.75], margin=0.0, size=64)
    assert out.size == (64, 64)
    # The red box fills the width; top and bottom are padding.
    assert out.getpixel((32, 32)) == (255, 0, 0)
    assert out.getpixel((32, 1)) != (0, 0, 255)
    assert embed.crop_for_model(img, None, size=32).size == (32, 32)


def test_embed_rows_caches(tmp_path: Path) -> None:
    images = tmp_path / "processed" / "images"
    rows = []
    for i in range(5):
        s = store_image(_png(i), images)
        rows.append(_row(s.image_id, "ragdoll", "train", "oxford", "cat"))
    calls: list[int] = []

    def fake(batch: list[Image.Image]) -> np.ndarray:
        calls.append(len(batch))
        return np.ones((len(batch), 4), dtype=np.float32)

    first = embed.embed_rows(rows, tmp_path, fake, "fake", batch_size=2)
    assert len(first) == 5 and calls == [2, 2, 1]
    again = embed.embed_rows(rows, tmp_path, fake, "fake", batch_size=2)
    assert calls == [2, 2, 1]  # nothing recomputed
    assert all(np.array_equal(first[k], again[k]) for k in first)


def _png(seed: int) -> bytes:
    import io

    arr = np.random.default_rng(seed).integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()
