"""Train the baseline per-species heads on frozen embeddings and report metrics.

v0 trains on Oxford plus the Commons domestic cats. Commons pedigree images are only
candidates until the screening step admits them, so they are not trained on yet; their
held-out groups are the unseen-source test.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from petlens.data.catalog import BreedCatalog, Species
from petlens.data.manifest import ManifestRow
from petlens.models.heads import SpeciesHead, evaluate, train_species_head
from petlens.models.metrics import FloatArray, IntArray, macro_f1, top_k_accuracy

TEST_SETS = ("test_in_domain", "test_unseen_source", "test_domestic")
LEARNING_CURVE = (0.25, 0.5, 1.0)


POOL = ("train", "val", "calibration")
LEAK_COSINE = 0.97  # held-out images this close to a training image are left out of evaluation


def in_v0_training_pool(r: ManifestRow) -> bool:
    """Oxford, or the domestic class from Commons. Unscreened Commons breeds are not."""
    return r.source == "oxford" or r.breed == "domestic_cat"


def in_training_pool(r: ManifestRow, admitted: frozenset[str]) -> bool:
    """v0 pool plus the Commons images screening admitted."""
    return in_v0_training_pool(r) or r.image_id in admitted


@dataclass
class SpeciesData:
    classes: list[str]
    x: dict[str, FloatArray]
    y: dict[str, IntArray]


def build_arrays(
    rows: list[ManifestRow],
    emb: dict[str, np.ndarray],
    breeds: BreedCatalog,
    species: Species,
    admitted: frozenset[str] = frozenset(),
) -> SpeciesData:
    classes = breeds.training_classes(species)
    index = {c: i for i, c in enumerate(classes)}
    buckets: dict[str, list[ManifestRow]] = {}
    for r in rows:
        if r.species != species or r.breed not in index or r.image_id not in emb:
            continue
        split = r.split or ""
        if split in POOL and not in_training_pool(r, admitted):
            continue
        if split in ("train", "val", "calibration", *TEST_SETS):
            buckets.setdefault(split, []).append(r)
    dim = len(next(iter(emb.values()))) if emb else 0
    x = {
        s: np.stack([emb[r.image_id] for r in rs]).astype(np.float64) if rs else np.zeros((0, dim))
        for s, rs in buckets.items()
    }
    y = {s: np.array([index[r.breed] for r in rs], dtype=np.int64) for s, rs in buckets.items()}
    return SpeciesData(classes, x, y)


def _subsample(n: int, fraction: float, seed: int) -> IntArray:
    rng = np.random.default_rng(seed)
    k = max(1, round(n * fraction))
    return np.sort(rng.permutation(n)[:k]).astype(np.int64)


def drop_leaks(d: SpeciesData, threshold: float = LEAK_COSINE) -> dict[str, int]:
    """Remove held-out images that are near-copies of a training image (by embedding).

    Splits never move; a leaking image is only left out of evaluation. Embeddings are
    L2-normalised, so the dot product is the cosine similarity.
    """
    dropped: dict[str, int] = {}
    if "train" not in d.x or not len(d.x["train"]):
        return dropped
    train_x = d.x["train"] / np.linalg.norm(d.x["train"], axis=1, keepdims=True)
    for split in TEST_SETS:
        if split not in d.x or not len(d.x[split]):
            continue
        x = d.x[split] / np.linalg.norm(d.x[split], axis=1, keepdims=True)
        keep = np.ones(len(x), dtype=bool)
        for start in range(0, len(x), 1024):
            sims = x[start : start + 1024] @ train_x.T
            keep[start : start + 1024] = sims.max(axis=1) < threshold
        dropped[split] = int((~keep).sum())
        d.x[split], d.y[split] = d.x[split][keep], d.y[split][keep]
    return dropped


def train(
    rows: list[ManifestRow],
    emb: dict[str, np.ndarray],
    breeds: BreedCatalog,
    seed: int = 0,
    admitted: frozenset[str] = frozenset(),
) -> tuple[dict[str, SpeciesHead], dict[str, object]]:
    heads: dict[str, SpeciesHead] = {}
    metrics: dict[str, object] = {}
    empty_x, empty_y = np.zeros((0, 1)), np.zeros(0, dtype=np.int64)
    for species in ("cat", "dog"):
        d = build_arrays(rows, emb, breeds, species, admitted)
        leaks = drop_leaks(d)
        if "train" not in d.x or len(set(d.y["train"].tolist())) < 2:
            continue

        def get(split: str, d: SpeciesData = d) -> tuple[FloatArray, IntArray]:
            return d.x.get(split, empty_x), d.y.get(split, empty_y)

        head = train_species_head(
            species, d.classes, *get("train"), *get("val"), *get("calibration"), seed=seed
        )
        heads[species] = head
        per_test = {}
        for test in TEST_SETS:
            x, y = get(test)
            if len(y):
                per_test[test] = evaluate(head, x, y)
        curve = []
        xt, yt = get("train")
        xi, yi = get("test_in_domain")
        for frac in LEARNING_CURVE:
            idx = _subsample(len(yt), frac, seed)
            if len(set(yt[idx].tolist())) < 2 or not len(yi):
                continue
            h = train_species_head(
                species,
                d.classes,
                xt[idx],
                yt[idx],
                *get("val"),
                *get("calibration"),
                seed=seed,
                kinds=("logreg",),
            )
            p = h.predict_proba(xi)
            curve.append(
                {
                    "fraction": frac,
                    "n_train": len(idx),
                    "top1": top_k_accuracy(p, yi, 1),
                    "macro_f1": macro_f1(yi, p.argmax(axis=1).astype(np.int64), len(d.classes)),
                }
            )
        metrics[species] = {
            "classes": d.classes,
            "head": head.kind,
            "val_macro_f1": head.candidates,
            "temperature": head.temperature,
            "split_sizes": {s: len(v) for s, v in d.y.items()},
            "admitted_commons": sum(
                1
                for r in rows
                if r.species == species and r.image_id in admitted and (r.split or "") in POOL
            ),
            "leaks_dropped": leaks,
            "tests": per_test,
            "learning_curve": curve,
        }
        if species == "cat" and "test_domestic" in d.y:
            p = head.predict_proba(d.x["test_domestic"])
            dom = d.classes.index("domestic_cat")
            metrics[species]["domestic_test_called_a_breed"] = float(  # type: ignore[index]
                np.mean(p.argmax(axis=1) != dom)
            )
    return heads, metrics


def render_report(run_id: str, dataset_version: str, model: str, metrics: dict[str, object]) -> str:
    lines = [
        f"# Baseline run {run_id}",
        "",
        f"- Dataset: {dataset_version}",
        f"- Embeddings: {model} (frozen)",
        "- Training pool: Oxford + Commons domestic cats + Commons breed photos admitted by "
        "screening",
        "",
    ]
    for species, m in metrics.items():
        assert isinstance(m, dict)
        lines += [
            f"## {species.capitalize()} head",
            "",
            f"- Classes: {len(m['classes'])}. Head picked on validation macro-F1: **{m['head']}** "
            f"({', '.join(f'{k} {v:.3f}' for k, v in m['val_macro_f1'].items())})",
            f"- Temperature: {m['temperature']:.3f}",
            f"- Commons breed photos admitted by screening (train/val/calibration): "
            f"{m['admitted_commons']}",
            f"- Held-out images left out as near-copies of training images: "
            f"{sum(m['leaks_dropped'].values())} {m['leaks_dropped'] or ''}",
            f"- Split sizes: {', '.join(f'{k} {v}' for k, v in sorted(m['split_sizes'].items()))}",
        ]
        if "domestic_test_called_a_breed" in m:
            lines.append(
                f"- Domestic test cats given a breed as top-1 (before any abstain policy): "
                f"{m['domestic_test_called_a_breed']:.1%}"
            )
        lines += [
            "",
            "| Test set | n | Top-1 | Top-3 | Macro-F1 | ECE before | ECE after |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for test, t in m["tests"].items():
            u, c = t["uncalibrated"], t["calibrated"]
            lines.append(
                f"| {test} | {int(c['n'])} | {c['top1']:.3f} | {c['top3']:.3f} | "
                f"{c['macro_f1']:.3f} | {u['ece']:.3f} | {c['ece']:.3f} |"
            )
        if m["learning_curve"]:
            lines += ["", "Learning curve (logistic regression, in-domain test):", ""]
            lines += [
                "| Train fraction | Images | Top-1 | Macro-F1 |",
                "| ---: | ---: | ---: | ---: |",
            ]
            for pt in m["learning_curve"]:
                lines.append(
                    f"| {pt['fraction']:.0%} | {pt['n_train']} | {pt['top1']:.3f} | "
                    f"{pt['macro_f1']:.3f} |"
                )
        lines.append("")
    return "\n".join(lines)


def save_run(
    models_dir: Path,
    dataset_version: str,
    model: str,
    heads: dict[str, SpeciesHead],
    metrics: dict[str, object],
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{dataset_version}"
    out = models_dir / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(heads, out / "heads.joblib")
    weights_hash = hashlib.sha256((out / "heads.joblib").read_bytes()).hexdigest()
    info = {
        "run_id": run_id,
        "dataset_version": dataset_version,
        "embedding_model": model,
        "weights_sha256": weights_hash,
        "metrics": metrics,
    }
    (out / "metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out / "report.md").write_text(render_report(run_id, dataset_version, model, metrics), "utf-8")
    logging.getLogger(__name__).info("saved run %s", out)
    return out
