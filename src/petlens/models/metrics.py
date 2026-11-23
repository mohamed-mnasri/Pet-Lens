"""Classification metrics used across training and evaluation (NumPy only)."""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def softmax(logits: FloatArray, temperature: float = 1.0) -> FloatArray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: FloatArray = e / e.sum(axis=1, keepdims=True)
    return out


def top_k_accuracy(probs: FloatArray, y: IntArray, k: int) -> float:
    top = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean(np.any(top == y[:, None], axis=1)))


def macro_f1(y_true: IntArray, y_pred: IntArray, n_classes: int) -> float:
    scores = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp + fn == 0:
            continue  # class absent from both: not counted
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else 0.0


def expected_calibration_error(probs: FloatArray, y: IntArray, n_bins: int = 15) -> float:
    """ECE with equal-width confidence bins on the top-1 probability."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def nll(probs: FloatArray, y: IntArray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None))))


def summary(probs: FloatArray, y: IntArray, n_classes: int) -> dict[str, float]:
    return {
        "n": float(len(y)),
        "top1": top_k_accuracy(probs, y, 1),
        "top3": top_k_accuracy(probs, y, 3),
        "macro_f1": macro_f1(y, probs.argmax(axis=1).astype(np.int64), n_classes),
        "ece": expected_calibration_error(probs, y),
        "nll": nll(probs, y),
    }
