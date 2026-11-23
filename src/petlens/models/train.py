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
from petlens.data.quality import is_weak
from petlens.models import policy
from petlens.models.heads import SpeciesHead, evaluate, train_species_head
from petlens.models.metrics import FloatArray, IntArray, macro_f1, top_k_accuracy
from petlens.models.openset import KnnIndex

TEST_SETS = ("test_in_domain", "test_unseen_source", "test_domestic")
POLICY_SETS = (*TEST_SETS, "test_unknown")
LEARNING_CURVE = (0.25, 0.5, 1.0)


POOL = ("train", "val", "calibration")
LEAK_COSINE = 0.97  # held-out images this close to a training image are left out of evaluation


def in_v0_training_pool(r: ManifestRow) -> bool:
    """Oxford, or the domestic class from Commons. Unscreened Commons breeds are not."""
    return r.source == "oxford" or r.breed == "domestic_cat"


NEVER_TRAIN_FLAGS = {"review_unsure", "audit_only"}


def in_training_pool(r: ManifestRow, admitted: frozenset[str]) -> bool:
    """v0 pool, the Commons images screening admitted, and labels I reviewed. Photos I
    wasn't sure about, blind-audit answers and weak photos (black and white, blurry, the
    animal cut off or tiny) are never trained on, unless I reviewed a weak one as fine."""
    if NEVER_TRAIN_FLAGS & set(r.flags) or is_weak(r):
        return False
    return in_v0_training_pool(r) or r.image_id in admitted or r.label_origin == "me"


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
        if r.species != species or r.image_id not in emb:
            continue
        split = r.split or ""
        if split == "test_unknown":
            buckets.setdefault(split, []).append(r)  # labelled -1: a breed the head doesn't know
            continue
        if r.breed not in index:
            continue
        if split in POOL and not in_training_pool(r, admitted):
            continue
        if split in (*POOL, "rw_calibration", *TEST_SETS):
            buckets.setdefault(split, []).append(r)
    dim = len(next(iter(emb.values()))) if emb else 0
    x = {
        s: np.stack([emb[r.image_id] for r in rs]).astype(np.float64) if rs else np.zeros((0, dim))
        for s, rs in buckets.items()
    }
    y = {
        s: np.array([index.get(r.breed, -1) for r in rs], dtype=np.int64)
        for s, rs in buckets.items()
    }
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
    for split in (*TEST_SETS, "rw_calibration", "test_unknown"):
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
    base_policy: policy.PolicyConfig | None = None,
) -> tuple[dict[str, SpeciesHead], dict[str, object]]:
    heads: dict[str, SpeciesHead] = {}
    metrics: dict[str, object] = {}
    cfg = (base_policy or policy.PolicyConfig()).model_copy(deep=True)
    close_pairs = [tuple(p) for p in breeds.close_pairs]
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
        metrics[species]["policy"] = _tune_and_evaluate_policy(  # type: ignore[index]
            d,
            head,
            species,
            cfg,
            close_pairs,  # type: ignore[arg-type]
        )
        if species == "cat" and "test_domestic" in d.y:
            p = head.predict_proba(d.x["test_domestic"])
            dom = d.classes.index("domestic_cat")
            metrics[species]["domestic_test_called_a_breed"] = float(  # type: ignore[index]
                np.mean(p.argmax(axis=1) != dom)
            )
    metrics["policy_config"] = cfg.model_dump()
    return heads, metrics


def _tune_and_evaluate_policy(
    d: SpeciesData,
    head: SpeciesHead,
    species: str,
    cfg: policy.PolicyConfig,
    close_pairs: list[tuple[str, str]],
) -> dict[str, object]:
    """Tune this species' thresholds into ``cfg`` and evaluate the policy on each test set."""
    index = KnnIndex(d.x["train"])
    cal_x = d.x.get("calibration")
    review = (
        float(np.quantile(index.distance(cal_x), 0.995))
        if cal_x is not None and len(cal_x)
        else 1.0
    )
    tune_split = "rw_calibration" if len(d.y.get("rw_calibration", [])) else "calibration"
    tx, ty = d.x[tune_split], d.y[tune_split]
    tprobs = head.predict_proba(tx)
    thresholds = policy.tune(
        tprobs, ty, index.distance(tx), d.classes, species, cfg, close_pairs, review
    )
    cfg.species[species] = thresholds
    cfg.tuned_on = "rw_calibration (cats), calibration split where no real-world slice exists"
    out: dict[str, object] = {
        "thresholds": thresholds.model_dump(),
        "tuned_on": tune_split,
        "tuned_on_n": len(ty),
    }
    domestic = d.classes.index(policy.DOMESTIC) if policy.DOMESTIC in d.classes else -1
    for split in POLICY_SETS:
        if split not in d.x or not len(d.y[split]):
            continue
        x, y = d.x[split], d.y[split]
        probs = head.predict_proba(x)
        knn = index.distance(x)
        ev = policy.evaluate(probs, y, knn, d.classes, species, cfg, close_pairs)
        entry: dict[str, object] = dict(ev.as_dict())
        if split == "test_unknown":
            entry["unknown_rejection_rate"] = 1 - ev.coverage
        else:
            entry["risk_coverage"] = policy.risk_coverage(probs, y)
        if split == "test_domestic" and domestic >= 0:
            entry["false_breed_claim_rate"] = ev.coverage  # any accept is a pedigree claim
        out[split] = entry
    return out


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
        if species == "policy_config":
            continue
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
        pol = m.get("policy")
        if pol:
            t = pol["thresholds"]
            lines += [
                "",
                f"Decision policy (tuned on {pol['tuned_on']}, n={pol['tuned_on_n']}): accept at "
                f"confidence >= {t['min_confidence']:.2f} and kNN distance <= "
                f"{t['max_knn_distance']:.3f}; review above {t['review_knn_distance']:.3f}.",
                "",
                "| Test set | n | Accepted | Accuracy when accepted | No clear breed | Review |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            for split in POLICY_SETS:
                e = pol.get(split)
                if not e:
                    continue
                lines.append(
                    f"| {split} | {int(e['n'])} | {e['coverage']:.1%} | "
                    f"{e['selective_accuracy']:.1%} | {e['no_clear_breed_rate']:.1%} | "
                    f"{e['review_rate']:.1%} |"
                )
            if "test_domestic" in pol:
                fbc = pol["test_domestic"]["false_breed_claim_rate"]
                lines.append("")
                lines.append(
                    f"- **False breed claim rate on domestic cats: {fbc:.1%}** (target 10%)"
                )
            if "test_unknown" in pol:
                rej = pol["test_unknown"]["unknown_rejection_rate"]
                lines.append(f"- Unknown-breed rejection rate: {rej:.1%}")
            for split in ("test_in_domain", "test_unseen_source"):
                rc = pol.get(split, {}).get("risk_coverage")
                if rc:
                    pts = ", ".join(f"{p['coverage']:.0%}: {p['risk']:.1%}" for p in rc)
                    lines.append(f"- Risk-coverage, {split} (coverage: error): {pts}")
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
