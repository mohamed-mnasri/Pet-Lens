"""Perceptual hashing (DCT pHash) and near-duplicate clustering.

Implemented with NumPy only, so it runs in CI without extra dependencies.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

HASH_SIZE = 8
_SAMPLE = 32


@lru_cache(maxsize=1)
def _dct_matrix(n: int = _SAMPLE) -> NDArray[np.float64]:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return np.asarray(m, dtype=np.float64)


def phash_image(img: Image.Image) -> int:
    """64-bit pHash: low-frequency DCT coefficients compared with their median."""
    gray = img.convert("L").resize((_SAMPLE, _SAMPLE), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    d = _dct_matrix()
    coeffs = d @ pixels @ d.T
    low = coeffs[:HASH_SIZE, :HASH_SIZE].flatten()
    bits = low > np.median(low[1:])  # skip the DC term when picking the threshold
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def phash_file(path: Path) -> str:
    with Image.open(path) as img:
        return f"{phash_image(img):016x}"


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Keep the smaller index as root so results don't depend on call order.
            lo, hi = min(ra, rb), max(ra, rb)
            self.parent[hi] = lo


def near_duplicate_pairs(hashes: Sequence[str], max_distance: int) -> list[tuple[int, int]]:
    """Index pairs ``(i, j)``, ``i < j``, whose hashes differ in ``max_distance`` bits or fewer."""
    if not hashes:
        return []
    values = np.array([int(h, 16) for h in hashes], dtype=np.uint64)
    pairs: list[tuple[int, int]] = []
    for i in range(len(values) - 1):
        dist = np.bitwise_count(values[i + 1 :] ^ values[i])
        for j in np.nonzero(dist <= max_distance)[0]:
            pairs.append((i, i + 1 + int(j)))
    return pairs
