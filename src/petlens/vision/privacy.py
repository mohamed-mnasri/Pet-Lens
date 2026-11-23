"""Blur people's faces before anything is stored.

COCO has no face class, so the face is approximated as the head region: the top of
each person box, as tall as the box is wide times ``head_ratio`` (about a seventh of a
standing person, under a third of a close-up), and the full box width. This keeps a pet held
in someone's arms visible while the face is hidden.
"""

from __future__ import annotations

from collections.abc import Iterable

from PIL import Image, ImageFilter

from petlens.vision.detect import Detection


def head_box(
    box: tuple[float, float, float, float], head_ratio: float = 0.45, max_frac: float = 0.3
) -> tuple[float, float, float, float]:
    """The head region of a person box (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = box
    head_h = min((y1 - y0) * max_frac, (x1 - x0) * head_ratio)
    return (x0, y0, x1, y0 + head_h)


def blur_people(
    img: Image.Image, detections: Iterable[Detection], pad: float = 0.05
) -> Image.Image:
    """Return a copy with the head region of every person (plus a margin) heavily blurred."""
    out = img.convert("RGB").copy()
    w, h = out.size
    for d in detections:
        if d.label != "person":
            continue
        x0, y0, x1, y1 = head_box(d.box)
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        box = (
            max(0, int(x0 - px)),
            max(0, int(y0 - py)),
            min(w, int(x1 + px + 0.5)),
            min(h, int(y1 + py + 0.5)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        region = out.crop(box)
        radius = max(12, max(region.size) // 8)
        # Downscale first so the blur is strong and fast even on large boxes.
        small = region.resize((max(1, region.width // 8), max(1, region.height // 8)))
        blurred = small.resize(region.size, Image.Resampling.BILINEAR).filter(
            ImageFilter.GaussianBlur(radius)
        )
        out.paste(blurred, box[:2])
    return out
