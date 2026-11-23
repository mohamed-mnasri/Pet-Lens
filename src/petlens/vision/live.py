"""Live camera (Step 16): the same per-frame pipeline, smoothed over the last few frames.

Each webcam frame goes through ``analyse_image`` like an uploaded photo. Animals are
followed from frame to frame by appearance (the same idea as clip grouping, no tracker
model): a detection joins the track of the same species whose recent embedding is most
similar, if it is similar enough, otherwise it starts a new track. The answer shown for
a track is the policy's decision on its probabilities and kNN distance averaged over its
last ``window`` frames, so one odd frame doesn't flip the label.

Storage follows the plan's limit: while an animal is in view, at most one frame every
``store_every`` seconds is kept (as an upload of kind ``live``); nothing is recorded.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from petlens.models.policy import Decision, PolicyConfig, decide
from petlens.vision.pipeline import AnimalResult


@dataclass(frozen=True)
class LiveConfig:
    window: int = 5
    min_similarity: float = 0.70
    max_missing: int = 8  # frames a track survives without being seen
    store_every: float = 10.0


@dataclass
class Track:
    id: int
    species: str
    embedding: NDArray[np.float64]
    probs: deque[NDArray[np.float64]]
    distances: deque[float]
    missing: int = 0


@dataclass
class Smoothed:
    track: int
    species: str
    box: tuple[float, float, float, float]
    decision: Decision
    frames: int  # how many frames the answer is averaged over


@dataclass
class LiveSession:
    """One browser tab's camera: its tracks, timing and storage clock."""

    policy: PolicyConfig
    classes: dict[str, list[str]]
    close_pairs: list[tuple[str, str]] = field(default_factory=list)
    cfg: LiveConfig = field(default_factory=LiveConfig)
    tracks: list[Track] = field(default_factory=list)
    last_stored: float = 0.0
    last_seen: float = field(default_factory=time.monotonic)
    frame_seconds: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    frame_times: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    _ids: itertools.count[int] = field(default_factory=itertools.count)

    def _unit(self, v: NDArray[np.float32]) -> NDArray[np.float64]:
        x = v.astype(np.float64)
        n = np.linalg.norm(x)
        return x / n if n else x

    def update(self, animals: list[AnimalResult]) -> list[Smoothed]:
        self.last_seen = time.monotonic()
        # perf_counter: time.monotonic() ticks every ~16 ms on Windows, so frames handled
        # within one tick all got the same time and fps() came out as 0.
        self.frame_times.append(time.perf_counter())
        free = list(self.tracks)
        out: list[Smoothed] = []
        # Most confident detections pick their track first.
        for a in sorted(animals, key=lambda a: -a.detection_score):
            v = self._unit(a.embedding)
            same = [t for t in free if t.species == a.species]
            best = max(same, key=lambda t: float(t.embedding @ v), default=None)
            if best is None or float(best.embedding @ v) < self.cfg.min_similarity:
                best = Track(
                    next(self._ids),
                    a.species,
                    v,
                    deque(maxlen=self.cfg.window),
                    deque(maxlen=self.cfg.window),
                )
                self.tracks.append(best)
            else:
                free.remove(best)
                mixed = 0.7 * best.embedding + 0.3 * v
                best.embedding = mixed / np.linalg.norm(mixed)
            best.missing = 0
            best.probs.append(a.probs)
            best.distances.append(a.knn_distance)
            probs = np.mean(best.probs, axis=0)
            dist = float(np.mean(best.distances))
            dec = decide(
                probs, self.classes[a.species], dist, a.species, self.policy, self.close_pairs
            )
            out.append(Smoothed(best.id, a.species, a.box, dec, len(best.probs)))
        for t in free:
            t.missing += 1
        self.tracks = [t for t in self.tracks if t.missing <= self.cfg.max_missing]
        return out

    def should_store(self, animals_in_view: bool) -> bool:
        now = time.monotonic()
        if animals_in_view and now - self.last_stored >= self.cfg.store_every:
            self.last_stored = now
            return True
        return False

    def fps(self) -> float:
        """Frames per second over the recent frames (what the browser actually gets)."""
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / span if span > 0 else 0.0


# --- the VLM and the camera never share the GPU ---------------------------------------

LIVE_MARKER = "live.active"
LIVE_TIMEOUT = 30.0


def mark_live(data_dir: Path) -> None:
    """Called on every camera frame: tells the VLM job to wait."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / LIVE_MARKER).write_text(str(time.time()), encoding="utf-8")


def live_active(data_dir: Path, timeout: float = LIVE_TIMEOUT) -> bool:
    path = data_dir / LIVE_MARKER
    try:
        return time.time() - float(path.read_text(encoding="utf-8")) < timeout
    except (OSError, ValueError):
        return False
