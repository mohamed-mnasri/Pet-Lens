"""Step 16: following animals across camera frames and smoothing their answers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from petlens.models.policy import PolicyConfig, SpeciesThresholds, decide
from petlens.vision import live
from petlens.vision.pipeline import AnimalResult

CLASSES = ["bengal", "domestic_cat", "ragdoll"]
CFG = PolicyConfig(
    species={
        "cat": SpeciesThresholds(min_confidence=0.6, max_knn_distance=0.5, review_knn_distance=0.9)
    }
)


def _animal(emb: np.ndarray, probs: list[float], score: float = 0.9) -> AnimalResult:
    p = np.array(probs)
    dec = decide(p, CLASSES, 0.2, "cat", CFG, [])
    return AnimalResult("cat", (10, 10, 50, 50), score, dec, p, 0.2, emb.astype(np.float32))


def test_one_odd_frame_does_not_flip_the_answer() -> None:
    rng = np.random.default_rng(0)
    milo, luna = rng.normal(size=16), rng.normal(size=16)
    s = live.LiveSession(CFG, {"cat": CLASSES})
    for _ in range(4):
        out = s.update([_animal(milo, [0.05, 0.05, 0.9])])
    odd = s.update([_animal(milo + rng.normal(size=16) * 0.05, [0.8, 0.1, 0.1])])
    assert odd[0].track == out[0].track and odd[0].decision.label == "ragdoll"
    assert odd[0].frames == 5
    both = s.update([_animal(milo, [0.05, 0.05, 0.9]), _animal(luna, [0.9, 0.05, 0.05], 0.8)])
    assert len({b.track for b in both}) == 2  # a second cat gets its own track
    for _ in range(live.LiveConfig().max_missing + 1):
        s.update([])
    assert s.tracks == []  # gone after enough empty frames
    assert s.fps() > 0


def test_storage_clock_and_live_marker(tmp_path: Path) -> None:
    s = live.LiveSession(CFG, {"cat": CLASSES}, cfg=live.LiveConfig(store_every=10))
    assert s.should_store(True) and not s.should_store(True)
    assert not s.should_store(False)
    assert not live.live_active(tmp_path)
    live.mark_live(tmp_path)
    assert live.live_active(tmp_path)
    (tmp_path / live.LIVE_MARKER).write_text(str(time.time() - 60))
    assert not live.live_active(tmp_path)
