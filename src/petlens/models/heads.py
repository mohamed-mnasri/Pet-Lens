"""Per-species classification heads on frozen embeddings.

Two candidates per species, a logistic regression and a small MLP, both from
scikit-learn. The one with the better validation macro-F1 wins, then a temperature is
fitted on the calibration split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from petlens.models.calibration import fit_temperature
from petlens.models.metrics import FloatArray, IntArray, macro_f1, softmax, summary

HeadKind = Literal["logreg", "mlp"]


def make_head(kind: HeadKind, seed: int) -> Pipeline:
    if kind == "logreg":
        clf: object = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(512,),
            alpha=1e-3,
            early_stopping=True,
            max_iter=300,
            random_state=seed,
        )
    return make_pipeline(StandardScaler(), clf)


def logits_of(model: Pipeline, x: FloatArray, n_classes: int) -> FloatArray:
    """Raw scores as an (n, n_classes) array, for both head kinds."""
    clf = model[-1]
    xs = model[:-1].transform(x)
    if isinstance(clf, LogisticRegression):
        z = np.asarray(clf.decision_function(xs), dtype=np.float64)
    else:
        # MLPClassifier has no decision_function; log-probabilities are logits up to a shift.
        z = np.log(np.clip(np.asarray(clf.predict_proba(xs), dtype=np.float64), 1e-12, None))
    full = np.full((len(x), n_classes), -1e9)
    full[:, np.asarray(clf.classes_, dtype=np.int64)] = z
    return full


@dataclass
class SpeciesHead:
    species: str
    classes: list[str]
    kind: HeadKind
    model: Pipeline
    temperature: float
    candidates: dict[str, float] = field(default_factory=dict)  # kind -> val macro-F1

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return softmax(logits_of(self.model, x, len(self.classes)), self.temperature)


def train_species_head(
    species: str,
    classes: list[str],
    x_train: FloatArray,
    y_train: IntArray,
    x_val: FloatArray,
    y_val: IntArray,
    x_cal: FloatArray,
    y_cal: IntArray,
    seed: int = 0,
    kinds: tuple[HeadKind, ...] = ("logreg", "mlp"),
) -> SpeciesHead:
    n = len(classes)
    best: tuple[float, HeadKind, Pipeline] | None = None
    scores: dict[str, float] = {}
    for kind in kinds:
        model = make_head(kind, seed).fit(x_train, y_train)
        f1 = macro_f1(y_val, logits_of(model, x_val, n).argmax(axis=1).astype(np.int64), n)
        scores[kind] = f1
        if best is None or f1 > best[0]:
            best = (f1, kind, model)
    assert best is not None
    _, kind, model = best
    temperature = fit_temperature(logits_of(model, x_cal, n), y_cal) if len(y_cal) else 1.0
    return SpeciesHead(species, classes, kind, model, temperature, scores)


def evaluate(head: SpeciesHead, x: FloatArray, y: IntArray) -> dict[str, dict[str, float]]:
    """Metrics before (T=1) and after temperature scaling."""
    z = logits_of(head.model, x, len(head.classes))
    n = len(head.classes)
    return {
        "uncalibrated": summary(softmax(z, 1.0), y, n),
        "calibrated": summary(softmax(z, head.temperature), y, n),
    }
