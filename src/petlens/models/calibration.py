"""Temperature scaling (Guo et al., 2017): one scalar fitted on the calibration split."""

from __future__ import annotations

import math

from petlens.models.metrics import FloatArray, IntArray, nll, softmax


def fit_temperature(
    logits: FloatArray, y: IntArray, lo: float = 0.05, hi: float = 20.0, iters: int = 80
) -> float:
    """Minimise NLL over T with a golden-section search in log space (NLL is unimodal in T)."""
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2

    def f(log_t: float) -> float:
        return nll(softmax(logits, math.exp(log_t)), y)

    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return math.exp((a + b) / 2)
