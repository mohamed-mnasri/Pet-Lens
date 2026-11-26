"""Live camera (Step 16): the same per-frame pipeline, smoothed over the last few frames.

Each webcam frame goes through ``analyse_image`` like an uploaded photo. Animals are
followed from frame to frame by appearance (the same idea as clip grouping, no tracker
model): a detection joins the track of the same species whose recent embedding is most
similar, if it is similar enough, otherwise it starts a new track. The answer shown for
a track is the policy's decision on its probabilities and kNN distance averaged over its
last ``window`` frames, so one odd frame doesn't flip the label.

Cats rarely hold still facing the camera, so once a breed's averaged probability on a
track reaches ``lock_at`` (75%), the track locks to that breed and keeps showing it until
the animal leaves the picture (its track ends after ``max_missing`` frames unseen). The
lock skips the kNN-distance check the photo policy uses: webcam frames sit far from the
training photos, so that check would keep a live answer on "no clear breed" for good. A
domestic-cat answer never locks, so a cat can still move on to a breed.

When exactly one animal of a species is in view and one track of that species is free,
the detection continues that track even if its appearance changed a lot (a cat turning
round), so a lock isn't lost to a new track.

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

from petlens.models.policy import DOMESTIC, Decision, PolicyConfig, decide
from petlens.vision.pipeline import AnimalResult


@dataclass(frozen=True)
class LiveConfig:
    window: int = 5
    min_similarity: float = 0.70
    max_missing: int = 8  # frames a track survives without being seen
    store_every: float = 10.0
    lock_at: float = 0.75  # averaged probability that locks a track to a breed
    lock_min_frames: int = 2  # not on a single frame


@dataclass
class Track:
    id: int
    species: str
    embedding: NDArray[np.float64]
    probs: deque[NDArray[np.float64]]
    distances: deque[float]
    missing: int = 0
    locked: str | None = None  # the breed this track is locked to
    locked_confidence: float = 0.0


@dataclass
class Smoothed:
    track: int
    species: str
    box: tuple[float, float, float, float]
    decision: Decision
    frames: int  # how many frames the answer is averaged over
    locked: bool = False


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
        per_species = {s: sum(1 for a in animals if a.species == s) for s in {"cat", "dog"}}
        for a in sorted(animals, key=lambda a: -a.detection_score):
            v = self._unit(a.embedding)
            same = [t for t in free if t.species == a.species]
            best = max(same, key=lambda t: float(t.embedding @ v), default=None)
            alone = per_species.get(a.species, 0) == 1 and len(same) == 1
            if best is None or (float(best.embedding @ v) < self.cfg.min_similarity and not alone):
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
            classes = self.classes[a.species]
            dec = decide(probs, classes, dist, a.species, self.policy, self.close_pairs)
            top = int(np.argmax(probs))
            if (
                best.locked is None
                and classes[top] != DOMESTIC
                and len(best.probs) >= self.cfg.lock_min_frames
                and float(probs[top]) >= self.cfg.lock_at
            ):
                best.locked = classes[top]
            if best.locked is not None:
                p = float(probs[classes.index(best.locked)])
                best.locked_confidence = max(best.locked_confidence, p)
                dec = Decision("accept", best.locked, best.locked_confidence, dist, dec.top3)
            out.append(
                Smoothed(best.id, a.species, a.box, dec, len(best.probs), best.locked is not None)
            )
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
