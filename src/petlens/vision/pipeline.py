"""Photo and clip analysis: detect, blur people, crop, embed, classify, decide.

``Analyzer`` holds the loaded pieces; ``analyse_image`` and ``analyse_clip`` are the
two entry points the API (Step 6) will call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from petlens.models.embed import EmbedFn, crop_for_model
from petlens.models.heads import SpeciesHead
from petlens.models.openset import KnnIndex
from petlens.models.policy import Decision, PolicyConfig, decide
from petlens.vision.detect import Detection, Detector
from petlens.vision.grouping import GroupingConfig, Item, group_items
from petlens.vision.privacy import blur_people
from petlens.vision.video import Frame


@dataclass
class Analyzer:
    detector: Detector
    embed: EmbedFn
    heads: dict[str, SpeciesHead]
    knn: dict[str, KnnIndex]
    policy: PolicyConfig
    close_pairs: list[tuple[str, str]] = field(default_factory=list)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)


@dataclass(frozen=True)
class AnimalResult:
    species: str
    box: tuple[float, float, float, float]
    detection_score: float
    decision: Decision
    probs: NDArray[np.float64]
    knn_distance: float
    embedding: NDArray[np.float32]


@dataclass(frozen=True)
class ImageResult:
    animals: list[AnimalResult]
    people_blurred: int
    image: Image.Image  # people blurred: the only version that may be stored
    seconds: float


@dataclass(frozen=True)
class GroupResult:
    species: str
    frames: list[int]
    decision: Decision


@dataclass(frozen=True)
class ClipResult:
    groups: list[GroupResult]
    frames: list[ImageResult]
    ambiguous: bool
    reason: str | None
    seconds: float


def _animals(a: Analyzer, img: Image.Image, dets: list[Detection]) -> list[AnimalResult]:
    animals = [d for d in dets if d.label in a.heads]
    if not animals:
        return []
    w, h = img.size
    crops = [crop_for_model(img, d.normalised(w, h)) for d in animals]
    embs = a.embed(crops)
    out = []
    for d, e in zip(animals, embs, strict=True):
        head = a.heads[d.label]
        x = e.astype(np.float64)[None, :]
        probs = head.predict_proba(x)[0]
        dist = float(a.knn[d.label].distance(x)[0])
        dec = decide(probs, head.classes, dist, d.label, a.policy, a.close_pairs)
        out.append(AnimalResult(d.label, d.box, d.score, dec, probs, dist, e))
    return out


def analyse_image(a: Analyzer, img: Image.Image) -> ImageResult:
    start = time.perf_counter()
    if img.mode != "RGB":
        img = img.convert("RGB")
    dets = a.detector(img)
    animals = _animals(a, img, dets)  # classify on the original pixels
    people = [d for d in dets if d.label == "person"]
    stored = blur_people(img, people) if people else img
    return ImageResult(animals, len(people), stored, time.perf_counter() - start)


def analyse_clip(a: Analyzer, frames: list[Frame]) -> ClipResult:
    start = time.perf_counter()
    per_frame = [analyse_image(a, f.image) for f in frames]
    items: list[Item] = []
    refs: list[AnimalResult] = []
    for f, res in zip(frames, per_frame, strict=True):
        for animal in res.animals:
            x0, y0, x1, y1 = animal.box
            items.append(Item(f.index, animal.species, animal.embedding, (x1 - x0) * (y1 - y0)))
            refs.append(animal)
    grouping = group_items(items, a.grouping)
    groups: list[GroupResult] = []
    if not grouping.ambiguous:
        for g in grouping.groups:
            members = [refs[i] for i in g.members]
            head = a.heads[g.species]
            probs = np.mean([m.probs for m in members], axis=0)
            dist = float(np.mean([m.knn_distance for m in members]))
            dec = decide(probs, head.classes, dist, g.species, a.policy, a.close_pairs)
            groups.append(GroupResult(g.species, sorted(g.frames), dec))
    return ClipResult(
        groups, per_frame, grouping.ambiguous, grouping.reason, time.perf_counter() - start
    )
