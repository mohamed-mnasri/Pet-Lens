"""Object detection for cats, dogs and people.

The real detector is RF-DETR (Apache-2.0 sizes only: nano, small, medium, base),
imported lazily from the optional ``ml`` extra. Everything downstream only sees
``Detection`` objects, so tests use a fake detector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    names: dict[int, str] = dict(getattr(model, "class_names", None) or _coco_names())
    min_threshold = min(cfg.animal_threshold, cfg.person_threshold)

    def detect(img: Image.Image) -> list[Detection]:
        found = model.predict(img.convert("RGB"), threshold=min_threshold)
        out: list[Detection] = []
        for box, score, class_id in zip(found.xyxy, found.confidence, found.class_id, strict=True):
            name = names.get(int(class_id), "")
            if name not in LABELS:
                continue
            threshold = cfg.person_threshold if name == "person" else cfg.animal_threshold
            if float(score) < threshold:
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            out.append(Detection(name, float(score), (x0, y0, x1, y1)))
        return out

    return detect


def _coco_names() -> dict[int, str]:
    try:
        from rfdetr.util.coco_classes import COCO_CLASSES

        return {int(k): str(v) for k, v in COCO_CLASSES.items()}
    except ImportError:  # COCO category ids, as used by RF-DETR's COCO checkpoints
        return {1: "person", 17: "cat", 18: "dog"}
