"""Metrics for one training run on its test sets, computed from the saved heads.

Per species and test set:

- headline: top-1, top-3, macro-F1, ECE (calibrated), with Wilson intervals on top-1
- decision policy: coverage, accuracy when accepted, no-clear-breed and review rates,
  the false breed claim rate on domestic cats and the unknown-breed rejection rate
- risk-coverage points (error among the most confident share)
- recall and precision per class, recall with a Wilson interval
- one row per test image (prediction, decision, flags), so the dbt marts can slice
  failures by source, breed, label origin or photo flag

Held-out images that are near-copies of a training image are left out, as in training.
The golden set (Step 11) is scored as its own test set, ``golden``, with its frozen labels.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from petlens.data.catalog import BreedCatalog
from petlens.data.manifest import ManifestRow
from petlens.models import policy
from petlens.models.golden import GoldenRow
from petlens.models.heads import SpeciesHead
from petlens.models.metrics import expected_calibration_error, macro_f1, top_k_accuracy
from petlens.models.openset import KnnIndex
from petlens.models.train import LEAK_COSINE, POLICY_SETS, in_training_pool

Z95 = 1.959964


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class EvalResult:
    metrics: list[dict[str, Any]] = field(default_factory=list)
    class_metrics: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)


def _unit(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return out


def evaluate(
    rows: list[ManifestRow],
    emb: dict[str, np.ndarray],
    heads: dict[str, SpeciesHead],
    breeds: BreedCatalog,
    cfg: policy.PolicyConfig,
    admitted: frozenset[str] = frozenset(),
    golden: list[GoldenRow] | None = None,
) -> EvalResult:
    out = EvalResult()
    by_id = {r.image_id: r for r in rows}
    gold = [
        replace(by_id[g.image_id], breed=g.breed, split="golden")
        for g in golden or []
        if g.image_id in by_id
    ]
    close_pairs = [(a, b) for a, b in breeds.close_pairs]
    for species, head in sorted(heads.items()):
        classes = head.classes
        index_of = {c: i for i, c in enumerate(classes)}
        mine = [r for r in rows if r.species == species and r.image_id in emb]
        train = [r for r in mine if r.split == "train" and in_training_pool(r, admitted)]
        if not train:
            continue
        train_x = np.stack([emb[r.image_id] for r in train]).astype(np.float64)
        knn = KnnIndex(train_x)
        train_unit = _unit(train_x)
        mine_gold = [r for r in gold if r.species == species and r.image_id in emb]
        for test in (*POLICY_SETS, "golden"):
            known = test != "test_unknown"
            pool = mine_gold if test == "golden" else mine
            sel = [r for r in pool if r.split == test and (not known or r.breed in index_of)]
            if not sel:
                continue
            x = np.stack([emb[r.image_id] for r in sel]).astype(np.float64)
            keep = (_unit(x) @ train_unit.T).max(axis=1) < LEAK_COSINE
            sel = [r for r, k in zip(sel, keep, strict=True) if k]
            x = x[keep]
            if not sel:
                continue
            y = np.array([index_of.get(r.breed, -1) for r in sel], dtype=np.int64)
            probs = head.predict_proba(x)
            dist = knn.distance(x)
            _add_set(out, species, test, sel, y, probs, dist, classes, cfg, close_pairs)
    return out


def _add_set(
    out: EvalResult,
    species: str,
    test: str,
    rows: list[ManifestRow],
    y: np.ndarray,
    probs: np.ndarray,
    dist: np.ndarray,
    classes: list[str],
    cfg: policy.PolicyConfig,
    close_pairs: list[tuple[str, str]],
) -> None:
    n = len(y)
    known = test != "test_unknown"

    def metric(name: str, value: float, ci: tuple[float, float] | None = None, **kw: Any) -> None:
        lo, hi = ci if ci else (None, None)
        out.metrics.append(
            {
                "species": species,
                "test_set": test,
                "metric": name,
                "param": kw.get("param"),
                "value": float(value),
                "lo": lo,
                "hi": hi,
                "n": int(kw.get("n", n)),
            }
        )

    pred = probs.argmax(axis=1)
    decisions = [
        policy.decide(p, classes, float(d), species, cfg, close_pairs)
        for p, d in zip(probs, dist, strict=True)
    ]
    accepted = [dec.outcome == "accept" for dec in decisions]
    if known:
        right = int((pred == y).sum())
        metric("top1", right / n, wilson(right, n))
        metric("top3", top_k_accuracy(probs, y, 3))
        metric("macro_f1", macro_f1(y, pred.astype(np.int64), len(classes)))
        metric("ece", expected_calibration_error(probs, y))
        conf = probs.max(axis=1)
        order = np.argsort(-conf)
        for share in (0.5, 0.8, 1.0):
            k = max(1, round(n * share))
            wrong = int((pred[order[:k]] != y[order[:k]]).sum())
            metric("risk_at_coverage", wrong / k, wilson(wrong, k), param=share, n=k)
        for c_idx, label in enumerate(classes):
            n_c = int((y == c_idx).sum())
            if not n_c:
                continue
            hit = int(((y == c_idx) & (pred == c_idx)).sum())
            called = int((pred == c_idx).sum())
            lo, hi = wilson(hit, n_c)
            out.class_metrics.append(
                {
                    "species": species,
                    "test_set": test,
                    "label": label,
                    "n": n_c,
                    "recall": hit / n_c,
                    "recall_lo": lo,
                    "recall_hi": hi,
                    "precision": hit / called if called else None,
                }
            )
    n_acc = sum(accepted)
    outcomes = Counter(dec.outcome for dec in decisions)
    metric("coverage", n_acc / n, wilson(n_acc, n))
    metric("no_clear_breed_rate", outcomes["no_clear_breed"] / n)
    metric("review_rate", outcomes["review"] / n)
    if known and n_acc:
        good = sum(
            1
            for dec, label, acc in zip(decisions, y, accepted, strict=True)
            if acc and dec.label == classes[label]
        )
        metric("selective_accuracy", good / n_acc, wilson(good, n_acc), n=n_acc)
    if test == "test_domestic":
        metric("false_breed_claim_rate", n_acc / n, wilson(n_acc, n))
    if test == "test_unknown":
        rejected = n - n_acc
        metric("unknown_rejection_rate", rejected / n, wilson(rejected, n))
    for r, label, p_idx, p, d, dec in zip(rows, y, pred, probs, dist, decisions, strict=True):
        out.images.append(
            {
                "image_id": r.image_id,
                "species": species,
                "test_set": test,
                "source": r.source,
                "label_origin": r.label_origin,
                "true_label": r.breed,
                "known_breed": bool(label >= 0),
                "predicted_label": classes[p_idx],
                "confidence": float(p[p_idx]),
                "knn_distance": float(d),
                "outcome": dec.outcome,
                "decided_label": dec.label,
                "top1_correct": bool(label >= 0 and p_idx == label),
                "flags": list(r.flags),
            }
        )


def composition(rows: list[ManifestRow]) -> list[dict[str, Any]]:
    """Image counts per source, species, breed, split and label origin."""
    counts = Counter((r.source, r.species, r.breed, r.split or "", r.label_origin) for r in rows)
    keys = ("source", "species", "breed", "split", "label_origin")
    return [dict(zip(keys, k, strict=True)) | {"n": n} for k, n in sorted(counts.items())]
