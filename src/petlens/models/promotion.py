"""Champion vs challenger: the promotion gate.

Both models answer the same photos, and the difference is judged with a paired
bootstrap (resampling photos, so each resample compares the two models on identical
images). A challenger is promoted only if all of these hold:

1. **Golden set:** top-1 accuracy is higher, with the gain above noise (the 5th
   percentile of the bootstrapped difference is above zero).
2. **Unseen-source test:** the same, on held-out photos from sources never trained on.
3. **No breed falls behind:** on golden and unseen-source photos together, no breed both
   models know (with at least ``min_breed_n`` photos) loses more than ``recall_margin``
   of recall.
4. **False breed claims:** on the domestic test cats, the share given a pedigree breed
   under each model's own decision policy does not go up.
5. **Golden stays clean:** no golden photo is in the challenger's training pool.

Photos either model has trained on, near-copies of either training pool (cosine 0.97 or
more, as in training) and photos missing from either embedding cache are left out, so
both are judged on exactly the same images.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from petlens.data.catalog import BreedCatalog
from petlens.models import golden as golden_set
from petlens.models import policy
from petlens.models.openset import KnnIndex
from petlens.models.runs import LoadedRun, Vectors, cached_vectors
from petlens.models.train import LEAK_COSINE

BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class GateConfig:
    n_boot: int = 10_000
    seed: int = 0
    alpha: float = 0.05  # one-sided: the gain must clear the 5th percentile
    recall_margin: float = 0.10
    min_breed_n: int = 10


@dataclass(frozen=True)
class Photo:
    image_id: str
    species: str
    breed: str


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    challenger: str
    champion: str | None
    checks: list[Check] = field(default_factory=list)
    numbers: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "challenger": self.challenger,
            "champion": self.champion,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
            "numbers": self.numbers,
        }

    def render(self) -> str:
        verdict = "PROMOTE" if self.passed else "KEEP THE CHAMPION"
        lines = [
            f"# Promotion gate: {verdict}",
            "",
            f"- Challenger: {self.challenger}",
            f"- Champion: {self.champion or '(none yet)'}",
            "",
            "| Check | Result | Detail |",
            "| --- | --- | --- |",
        ]
        for c in self.checks:
            lines.append(f"| {c.name} | {'pass' if c.passed else 'FAIL'} | {c.detail} |")
        return "\n".join(lines) + "\n"


def paired_bootstrap(
    a: BoolArray, b: BoolArray, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Difference in accuracy (b minus a) with its ``alpha`` and ``1 - alpha`` percentiles,
    resampling the same photos for both models."""
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    diff = b.astype(np.float64) - a.astype(np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 500):
        k = min(500, n_boot - start)
        idx = rng.integers(0, n, size=(k, n))
        means[start : start + k] = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(diff.mean()), float(lo), float(hi)


def _unit(x: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    return out


class Judge:
    """One run's answers on any photo in its embedding cache."""

    def __init__(self, run: LoadedRun, vecs: Vectors, close_pairs: list[tuple[str, str]]) -> None:
        self.run = run
        self.close_pairs = close_pairs
        self.vecs = vecs
        self.trained: set[str] = set()
        self.knn: dict[str, KnnIndex] = {}
        self.train_unit: dict[str, NDArray[np.float64]] = {}
        for species in run.heads:
            ids = [i for i in run.training_ids(species) if i in vecs]
            self.trained.update(ids)
            if ids:
                x = np.stack([vecs[i] for i in ids]).astype(np.float64)
                self.knn[species] = KnnIndex(x)
                self.train_unit[species] = _unit(x)

    def usable(self, p: Photo) -> bool:
        if p.image_id not in self.vecs or p.image_id in self.trained:
            return False
        ref = self.train_unit.get(p.species)
        if ref is None:
            return True
        v = _unit(self.vecs[p.image_id].astype(np.float64)[None, :])
        return bool((v @ ref.T).max() < LEAK_COSINE)

    def answers(self, photos: list[Photo]) -> tuple[list[str | None], list[str | None]]:
        """Top-1 label and the decision policy's outcome per photo (None: no head)."""
        top1: list[str | None] = [None] * len(photos)
        outcome: list[str | None] = [None] * len(photos)
        for species, head in self.run.heads.items():
            idx = [i for i, p in enumerate(photos) if p.species == species]
            if not idx:
                continue
            x = np.stack([self.vecs[photos[i].image_id] for i in idx]).astype(np.float64)
            probs = head.predict_proba(x)
            dist = self.knn[species].distance(x) if species in self.knn else np.zeros(len(idx))
            for j, i in enumerate(idx):
                top1[i] = head.classes[int(probs[j].argmax())]
                outcome[i] = policy.decide(
                    probs[j], head.classes, float(dist[j]), species, self.run.cfg, self.close_pairs
                ).outcome
        return top1, outcome


def _correct(judge: Judge, photos: list[Photo]) -> BoolArray:
    top1, _ = judge.answers(photos)
    return np.array([t == p.breed for t, p in zip(top1, photos, strict=True)], dtype=bool)


def _accuracy_check(
    name: str,
    photos: list[Photo],
    a: Judge,
    b: Judge,
    cfg: GateConfig,
    numbers: dict[str, Any],
) -> Check:
    if not photos:
        return Check(name, False, "no photos to compare on")
    ca, cb = _correct(a, photos), _correct(b, photos)
    diff, lo, hi = paired_bootstrap(ca, cb, cfg.n_boot, cfg.seed, cfg.alpha)
    numbers[name] = {
        "n": len(photos),
        "champion_top1": float(ca.mean()),
        "challenger_top1": float(cb.mean()),
        "diff": diff,
        "diff_lo": lo,
        "diff_hi": hi,
    }
    detail = (
        f"n={len(photos)}: {ca.mean():.1%} to {cb.mean():.1%} "
        f"(gain {diff:+.1%}, 90% range {lo:+.1%} to {hi:+.1%})"
    )
    return Check(name, lo > 0, detail)


def _recall_check(photos: list[Photo], a: Judge, b: Judge, cfg: GateConfig) -> Check:
    both = {
        (s, c)
        for s in a.run.heads
        if s in b.run.heads
        for c in set(a.run.heads[s].classes) & set(b.run.heads[s].classes)
    }
    ca, cb = _correct(a, photos), _correct(b, photos)
    drops = []
    for key in sorted(both):
        mask = np.array([(p.species, p.breed) == key for p in photos], dtype=bool)
        n = int(mask.sum())
        if n < cfg.min_breed_n:
            continue
        ra, rb = float(ca[mask].mean()), float(cb[mask].mean())
        if ra - rb > cfg.recall_margin:
            drops.append(f"{key[1]} {ra:.0%} to {rb:.0%} (n={n})")
    if drops:
        return Check("no breed falls behind", False, "; ".join(drops))
    return Check(
        "no breed falls behind",
        True,
        f"no breed with {cfg.min_breed_n}+ photos lost more than {cfg.recall_margin:.0%}",
    )


def _claim_check(photos: list[Photo], a: Judge, b: Judge, numbers: dict[str, Any]) -> Check:
    name = "false breed claims"
    if not photos:
        return Check(name, True, "no domestic test cats to compare on")
    rates = []
    for judge in (a, b):
        _, outcome = judge.answers(photos)
        rates.append(sum(o == "accept" for o in outcome) / len(photos))
    numbers[name] = {"n": len(photos), "champion": rates[0], "challenger": rates[1]}
    return Check(
        name,
        rates[1] <= rates[0],
        f"n={len(photos)} domestic cats: {rates[0]:.1%} to {rates[1]:.1%}",
    )


def _photos(run: LoadedRun, split: str) -> list[Photo]:
    return [
        Photo(r.image_id, r.species, r.breed)
        for r in run.rows
        if r.split == split and r.species in run.heads and r.breed in run.heads[r.species].classes
    ]


def compare(
    champion: LoadedRun | None,
    challenger: LoadedRun,
    golden: list[golden_set.GoldenRow],
    data_dir: Path,
    breeds: BreedCatalog,
    cfg: GateConfig = GateConfig(),
    vectors: dict[str, Vectors] | None = None,
) -> GateResult:
    """Run the gate. ``vectors`` maps an embedding model to its cache (tests pass it in)."""
    cache = dict(vectors or {})

    def vecs_for(run: LoadedRun) -> Vectors:
        if run.embedding_model not in cache:
            cache[run.embedding_model] = cached_vectors(run.embedding_model, data_dir)
        return cache[run.embedding_model]

    pairs = [(x, y) for x, y in breeds.close_pairs]
    b = Judge(challenger, vecs_for(challenger), pairs)
    result = GateResult(challenger.run_id, champion.run_id if champion else None)
    golden_ids = {g.image_id for g in golden}
    leaked = golden_ids & b.trained
    result.checks.append(
        Check(
            "golden set never trained on",
            not leaked,
            f"{len(leaked)} golden photos in the training pool" if leaked else "clean",
        )
    )
    if champion is None:
        result.checks.append(Check("champion", True, "none yet: the challenger becomes champion"))
        return result
    a = Judge(champion, vecs_for(champion), pairs)

    def shared(photos: list[Photo]) -> list[Photo]:
        return [p for p in photos if a.usable(p) and b.usable(p)]

    gold = shared([Photo(g.image_id, g.species, g.breed) for g in golden])
    unseen = shared(
        [p for p in _photos(challenger, "test_unseen_source") if p.image_id not in golden_ids]
    )
    domestic = shared(
        [p for p in _photos(challenger, "test_domestic") if p.breed == policy.DOMESTIC]
    )
    result.numbers["left_out_golden"] = len(golden) - len(gold)
    result.checks.append(_accuracy_check("golden set", gold, a, b, cfg, result.numbers))
    result.checks.append(_accuracy_check("unseen-source test", unseen, a, b, cfg, result.numbers))
    result.checks.append(_recall_check(gold + unseen, a, b, cfg))
    result.checks.append(_claim_check(domestic, a, b, result.numbers))
    return result
