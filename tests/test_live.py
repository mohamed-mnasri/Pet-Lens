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


def _far(emb: np.ndarray, probs: list[float]) -> AnimalResult:
    """A webcam-like frame: far from the training photos (kNN 0.7 > the 0.5 limit)."""
    p = np.array(probs)
    dec = decide(p, CLASSES, 0.7, "cat", CFG, [])
    return AnimalResult("cat", (10, 10, 50, 50), 0.9, dec, p, 0.7, emb.astype(np.float32))


def test_a_breed_at_75_percent_locks_until_the_cat_leaves() -> None:
    rng = np.random.default_rng(1)
    milo = rng.normal(size=16)
    s = live.LiveSession(CFG, {"cat": CLASSES})
    first = s.update([_far(milo, [0.1, 0.1, 0.8])])
    assert first[0].decision.outcome == "no_clear_breed" and not first[0].locked  # one frame
    locked = s.update([_far(milo, [0.1, 0.1, 0.8])])
    assert locked[0].locked and locked[0].decision.label == "ragdoll"
    assert locked[0].decision.outcome == "accept"  # despite the kNN distance
    # Turned away: the answer would drop, and its look changed a lot, but it stays.
    for _ in range(6):
        out = s.update([_far(rng.normal(size=16), [0.4, 0.4, 0.2])])
    assert out[0].track == locked[0].track and out[0].decision.label == "ragdoll"
    for _ in range(live.LiveConfig().max_missing + 1):
        s.update([])
    back = s.update([_far(milo, [0.4, 0.4, 0.2])])
    assert not back[0].locked  # it left the picture, so the lock is gone


def test_a_domestic_cat_answer_never_locks() -> None:
    s = live.LiveSession(CFG, {"cat": CLASSES})
    milo = np.ones(16)
    for _ in range(3):
        out = s.update([_far(milo, [0.05, 0.9, 0.05])])
    assert not out[0].locked


def test_storage_clock_and_live_marker(tmp_path: Path) -> None:
    s = live.LiveSession(CFG, {"cat": CLASSES}, cfg=live.LiveConfig(store_every=10))
    assert s.should_store(True) and not s.should_store(True)
    assert not s.should_store(False)
    assert not live.live_active(tmp_path)
    live.mark_live(tmp_path)
    assert live.live_active(tmp_path)
    (tmp_path / live.LIVE_MARKER).write_text(str(time.time() - 60))
    assert not live.live_active(tmp_path)
