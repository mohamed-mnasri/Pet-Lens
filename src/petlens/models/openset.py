"""Open-set score: how far a crop's embedding is from everything the head was trained on.

The score is ``1 - mean cosine similarity`` to the ``k`` nearest training embeddings of
the same species. Unseen breeds and odd inputs land further away than familiar ones.
This in-memory index has the same interface the pgvector-backed one will have in
Step 6, so the policy doesn't change when serving moves to Postgres.
"""

from __future__ import annotations

import numpy as np

from petlens.models.metrics import FloatArray

DEFAULT_K = 10


def _normalise(x: FloatArray) -> FloatArray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    out: FloatArray = x / np.clip(norms, 1e-12, None)
    return out


class KnnIndex:
    def __init__(self, train_x: FloatArray, k: int = DEFAULT_K) -> None:
        if not len(train_x):
            raise ValueError("empty training set")
        self.train = _normalise(np.asarray(train_x, dtype=np.float64))
        self.k = min(k, len(self.train))

    def distance(self, x: FloatArray, batch: int = 1024) -> FloatArray:
        x = _normalise(np.asarray(x, dtype=np.float64))
        out = np.empty(len(x))
        for start in range(0, len(x), batch):
            sims = x[start : start + batch] @ self.train.T
            top = np.partition(sims, -self.k, axis=1)[:, -self.k :]
            out[start : start + batch] = 1.0 - top.mean(axis=1)
        return out
