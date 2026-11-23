"""Decision policy: accept, no clear breed, or review.

- **accept**: "looks like X". Calibrated confidence and the open-set score both pass.
- **no_clear_breed**: for cats, "domestic cat, no clear breed", either because
  ``domestic_cat`` won or because no breed passed the thresholds while the crop still
  looks like the cats the model knows. For dogs, "no clear breed".
- **review**: unsure. The crop is far from anything in training (an odd input), so it
  goes to the review queue.

Close pairs: when the runner-up is the top breed's close-pair partner and their
probabilities are within ``close_margin``, an accepted answer names both.

Thresholds live in ``data/policy.yaml`` and are tuned on the real-world calibration
slice, never on the in-domain data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from petlens.models.metrics import FloatArray, IntArray

Outcome = Literal["accept", "no_clear_breed", "review"]
DOMESTIC = "domestic_cat"


class SpeciesThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_confidence: float = Field(0.6, ge=0, le=1)
    max_knn_distance: float = Field(0.5, ge=0, le=2)  # accept only at or below this
    review_knn_distance: float = Field(0.7, ge=0, le=2)  # review above this


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    close_margin: float = Field(0.15, ge=0, le=1)
    target_accuracy: float = Field(0.90, ge=0, le=1)  # accepted answers must be this accurate
    max_false_breed_claim: float = Field(0.10, ge=0, le=1)
    species: dict[str, SpeciesThresholds] = Field(
        default_factory=lambda: {"cat": SpeciesThresholds(), "dog": SpeciesThresholds()}
    )
    tuned_on: str = ""


def load(path: Path) -> PolicyConfig:
    with path.open(encoding="utf-8") as fh:
        return PolicyConfig.model_validate(yaml.safe_load(fh))


def save(cfg: PolicyConfig, path: Path) -> None:
    header = "# Decision policy thresholds. Written by `make train` (tuned on rw_calibration).\n"
    body = yaml.safe_dump(cfg.model_dump(), sort_keys=False)
    path.write_text(header + body, encoding="utf-8")


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    label: str | None  # the breed for accept; domestic_cat for a cat's no_clear_breed
    confidence: float
    knn_distance: float
    top3: list[tuple[str, float]]
    also: str | None = None  # close-pair partner named alongside an accepted breed


def decide(
    probs: FloatArray,
    classes: list[str],
    knn_distance: float,
    species: str,
    cfg: PolicyConfig,
    close_pairs: list[tuple[str, str]],
) -> Decision:
    t = cfg.species.get(species, SpeciesThresholds())
    order = np.argsort(-probs)
    top3 = [(classes[i], float(probs[i])) for i in order[:3]]
    best, conf = top3[0]
    no_breed_label = DOMESTIC if species == "cat" else None

    if knn_distance > t.review_knn_distance:
        return Decision("review", None, conf, knn_distance, top3)
    if best == DOMESTIC:
        return Decision("no_clear_breed", DOMESTIC, conf, knn_distance, top3)
    if conf < t.min_confidence or knn_distance > t.max_knn_distance:
        return Decision("no_clear_breed", no_breed_label, conf, knn_distance, top3)
    also = None
    if len(top3) > 1:
        runner, p2 = top3[1]
        if conf - p2 <= cfg.close_margin and (
            (best, runner) in close_pairs or (runner, best) in close_pairs
        ):
            also = runner
    return Decision("accept", best, conf, knn_distance, top3, also)


# --------------------------------------------------------------------------- tuning and evaluation


@dataclass
class PolicyEval:
    n: int = 0
    accepted: int = 0
    accepted_correct: int = 0
    no_clear_breed: int = 0
    review: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return self.accepted / self.n if self.n else 0.0

    @property
    def selective_accuracy(self) -> float:
        return self.accepted_correct / self.accepted if self.accepted else 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "n": self.n,
            "coverage": self.coverage,
            "selective_accuracy": self.selective_accuracy,
            "no_clear_breed_rate": self.no_clear_breed / self.n if self.n else 0.0,
            "review_rate": self.review / self.n if self.n else 0.0,
        }


def evaluate(
    probs: FloatArray,
    y: IntArray,
    knn: FloatArray,
    classes: list[str],
    species: str,
    cfg: PolicyConfig,
    close_pairs: list[tuple[str, str]],
) -> PolicyEval:
    """``y`` may be -1 for images of breeds the model doesn't know."""
    ev = PolicyEval(n=len(y))
    for p, label, d in zip(probs, y, knn, strict=True):
        dec = decide(p, classes, float(d), species, cfg, close_pairs)
        if dec.outcome == "accept":
            ev.accepted += 1
            ev.accepted_correct += int(label >= 0 and dec.label == classes[label])
        elif dec.outcome == "no_clear_breed":
            ev.no_clear_breed += 1
        else:
            ev.review += 1
    return ev


def risk_coverage(probs: FloatArray, y: IntArray, points: int = 10) -> list[dict[str, float]]:
    """Error rate among the most confident ``coverage`` share of images."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    order = np.argsort(-conf)
    out = []
    for i in range(1, points + 1):
        k = max(1, round(len(order) * i / points))
        out.append({"coverage": k / len(order), "risk": float(1 - correct[order[:k]].mean())})
    return out


def tune(
    probs: FloatArray,
    y: IntArray,
    knn: FloatArray,
    classes: list[str],
    species: str,
    base: PolicyConfig,
    close_pairs: list[tuple[str, str]],
    review_knn_distance: float,
) -> SpeciesThresholds:
    """Pick thresholds that accept the most real-world images while keeping accepted
    answers at ``target_accuracy`` and the false breed claim rate on domestic cats at
    or under ``max_false_breed_claim``. Falls back to the strictest grid point."""
    domestic = classes.index(DOMESTIC) if DOMESTIC in classes else -1
    conf_grid = np.round(np.arange(0.30, 0.96, 0.05), 2)
    knn_grid = np.unique(np.quantile(knn, np.linspace(0.5, 1.0, 11))) if len(knn) else [1.0]
    best: tuple[int, float, float] | None = None
    strictest = SpeciesThresholds(
        min_confidence=float(conf_grid[-1]),
        max_knn_distance=float(min(knn_grid)),
        review_knn_distance=review_knn_distance,
    )
    for c in conf_grid:
        for k in knn_grid:
            t = SpeciesThresholds(
                min_confidence=float(c),
                max_knn_distance=float(min(k, review_knn_distance)),
                review_knn_distance=review_knn_distance,
            )
            cfg = base.model_copy(update={"species": {**base.species, species: t}})
            ev = evaluate(probs, y, knn, classes, species, cfg, close_pairs)
            # Accepted domestic cats: a pedigree was claimed for a cat without one.
            claims = 0
            n_dom = int((y == domestic).sum())
            if n_dom:
                dom = y == domestic
                claims = evaluate(
                    probs[dom], y[dom], knn[dom], classes, species, cfg, close_pairs
                ).accepted
            fbc = claims / n_dom if n_dom else 0.0
            if ev.selective_accuracy < base.target_accuracy or fbc > base.max_false_breed_claim:
                continue
            key = (ev.accepted, -float(c), -float(k))
            if best is None or key > (best[0], -best[1], -best[2]):
                best = (ev.accepted, float(c), float(k))
    if best is None:
        return strictest
    return SpeciesThresholds(
        min_confidence=best[1],
        max_knn_distance=min(best[2], review_knn_distance),
        review_knn_distance=review_knn_distance,
    )
