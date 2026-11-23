"""Object detection for cats, dogs and people.

The real detector is RF-DETR (Apache-2.0 sizes only: nano, small, medium, base),
imported lazily from the optional ``ml`` extra. Everything downstream only sees
``Detection`` objects, so tests use a fake detector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from PIL import Image

Label = Literal["cat", "dog", "person"]
LABELS: tuple[Label, ...] = ("cat", "dog", "person")
APACHE_SIZES = ("nano", "small", "medium", "base")


@dataclass(frozen=True)
class Detection:
    label: Label
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    # Other animal labels the detector gave the same box (e.g. "dog" for a cat it wasn't
    # sure about). The pipeline settles these with the embedding.
    alternatives: tuple[str, ...] = ()

    @property
    def area(self) -> float:
        x0, y0, x1, y1 = self.box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def normalised(self, width: int, height: int) -> list[float]:
        x0, y0, x1, y1 = self.box
        return [x0 / width, y0 / height, x1 / width, y1 / height]


Detector = Callable[[Image.Image], list[Detection]]


@dataclass(frozen=True)
class DetectorConfig:
    size: str = "base"
    animal_threshold: float = 0.5
    # People are blurred, so a lower threshold errs on the side of privacy.
    person_threshold: float = 0.3
    # A second animal label on (nearly) the same box is kept as an alternative if it
    # scores at least this much; boxes this overlapping count as the same animal.
    alternative_threshold: float = 0.3
    same_animal_iou: float = 0.7


def load_rfdetr(cfg: DetectorConfig = DetectorConfig()) -> Detector:
    """Load an RF-DETR model trained on COCO and wrap it as a ``Detector``."""
    if cfg.size not in APACHE_SIZES:
        raise ValueError(f"RF-DETR size must be one of {APACHE_SIZES} (Apache-2.0 licensed)")
    import rfdetr

    cls = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "base": "RFDETRBase",
    }[cfg.size]
    model: Any = getattr(rfdetr, cls)()
    min_threshold = min(cfg.animal_threshold, cfg.person_threshold, cfg.alternative_threshold)

    def detect(img: Image.Image) -> list[Detection]:
        found = model.predict(img.convert("RGB"), threshold=min_threshold)
        names = class_names_of(found)
        people: list[Detection] = []
        animals: list[Detection] = []
        for box, score, name in zip(found.xyxy, found.confidence, names, strict=True):
            if name not in LABELS:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            det = Detection(name, float(score), (x0, y0, x1, y1))
            if name == "person":
                if det.score >= cfg.person_threshold:
                    people.append(det)
            else:
                animals.append(det)
        return merge_animals(animals, cfg) + people

    return detect


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_animals(
    cands: list[Detection], cfg: DetectorConfig = DetectorConfig()
) -> list[Detection]:
    """One detection per animal.

    RF-DETR scores classes independently, so the same box can come back as both "cat"
    and "dog" (and as near-duplicates of one label). The strongest box above the animal
    threshold wins; an overlapping box with another label and at least
    ``alternative_threshold`` becomes one of its ``alternatives``; other overlaps go.
    """
    kept: list[Detection] = []
    for d in sorted(cands, key=lambda d: d.score, reverse=True):
        match = next(
            (i for i, k in enumerate(kept) if iou(k.box, d.box) >= cfg.same_animal_iou), None
        )
        if match is None:
            if d.score >= cfg.animal_threshold:
                kept.append(d)
            continue
        k = kept[match]
        if (
            d.label != k.label
            and d.label not in k.alternatives
            and d.score >= cfg.alternative_threshold
        ):
            kept[match] = replace(k, alternatives=(*k.alternatives, d.label))
    return kept


# COCO category ids (1-based, with gaps), which is what RF-DETR's COCO checkpoints put in
# ``class_id``. Note ``model.class_names`` is a 0-based list of the 80 names, so indexing
# it with ``class_id`` is off (17 would read "horse" instead of "cat").
COCO_IDS: dict[int, str] = {1: "person", 17: "cat", 18: "dog"}


def class_names_of(found: Any) -> list[str]:
    """Class names for each detection: ``data["class_name"]`` if present, else COCO ids."""
    data = getattr(found, "data", None) or {}
    if "class_name" in data:
        return [str(n) for n in data["class_name"]]
    return [COCO_IDS.get(int(c), "") for c in found.class_id]
