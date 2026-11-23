"""Photo checks that keep weak photos out of review and training.

Swiping showed what slows review down and teaches the model little: black-and-white
photos, blurry ones, close-ups where only part of the animal shows (paws, a nose), and
animals that are a speck in the corner. Each gets a flag:

- ``grayscale``: almost no colour anywhere (breed colour is a big cue).
- ``blurry``: little fine detail (low variance of the Laplacian at a fixed size).
- ``cut_off``: the animal's box touches all four edges, so part of it is outside.
- ``small_animal``: the box covers under 4% of the photo.
- ``no_animal``: the detector found no cat or dog at all (paws only, a toy, a drawing).

Flagged photos still sit in the manifest and keep their split, so test sets stay
stable. They just aren't trained on or queued for review, unless I reviewed them and
said they were fine. ``python -m petlens.data.quality`` (``make data-quality``) runs the
detector over photos with no box yet; ``make data-build`` applies the flags.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from petlens.data.imaging import stored_path
from petlens.data.manifest import ManifestRow

log = logging.getLogger(__name__)

QUALITY_FLAGS = frozenset({"grayscale", "blurry", "cut_off", "small_animal", "no_animal"})
CACHE_VERSION = 1


@dataclass(frozen=True)
class QualityPolicy:
    check_size: int = 512  # long edge the pixel checks run at, so stored size doesn't matter
    grayscale_spread: float = 6.0  # mean per-pixel max-min channel difference, 0..255
    blur_variance: float = 15.0  # Laplacian variance below this counts as blurry
    edge_margin: float = 0.01
    min_box_area: float = 0.04


DEFAULT_QUALITY = QualityPolicy()


def _small(img: Image.Image, size: int) -> Image.Image:
    small = img.convert("RGB")
    small.thumbnail((size, size), Image.Resampling.BILINEAR)
    return small


def colour_spread(img: Image.Image, policy: QualityPolicy = DEFAULT_QUALITY) -> float:
    a = np.asarray(_small(img, policy.check_size), dtype=np.int16)
    return float((a.max(axis=2) - a.min(axis=2)).mean())


def sharpness(img: Image.Image, policy: QualityPolicy = DEFAULT_QUALITY) -> float:
    g = np.asarray(_small(img, policy.check_size).convert("L"), dtype=np.float32)
    if min(g.shape) < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def pixel_flags(img: Image.Image, policy: QualityPolicy = DEFAULT_QUALITY) -> list[str]:
    flags = []
    if colour_spread(img, policy) < policy.grayscale_spread:
        flags.append("grayscale")
    if sharpness(img, policy) < policy.blur_variance:
        flags.append("blurry")
    return flags


def box_flags(box: Sequence[float] | None, policy: QualityPolicy = DEFAULT_QUALITY) -> list[str]:
    """Flags for a normalised ``[x0, y0, x1, y1]`` box. ``None`` means none was found."""
    if box is None:
        return ["no_animal"]
    x0, y0, x1, y1 = box
    m = policy.edge_margin
    flags = []
    if x0 <= m and y0 <= m and x1 >= 1 - m and y1 >= 1 - m:
        flags.append("cut_off")
    if max(0.0, x1 - x0) * max(0.0, y1 - y0) < policy.min_box_area:
        flags.append("small_animal")
    return flags


# --- the cache in data/processed -----------------------------------------------------
# {"version": 1, "checks": {image_id: {"pixel": [...], "box": [..] | null, "detected": bool}}}
# Raw results, not flags, so thresholds can change without rerunning anything.

BoxFinder = Callable[[Image.Image], list[float] | None]


def cache_path(data_dir: Path) -> Path:
    return data_dir / "processed" / "photo_checks.json"


def read_cache(data_dir: Path) -> dict[str, dict[str, Any]]:
    path = cache_path(data_dir)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("version") != CACHE_VERSION:
        return {}
    checks: dict[str, dict[str, Any]] = raw["checks"]
    return checks


def write_cache(data_dir: Path, checks: dict[str, dict[str, Any]]) -> None:
    path = cache_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {"version": CACHE_VERSION, "checks": checks}
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def check_photos(
    rows: Iterable[ManifestRow],
    data_dir: Path,
    find_box: BoxFinder | None = None,
    policy: QualityPolicy = DEFAULT_QUALITY,
) -> dict[str, dict[str, Any]]:
    """Fill the cache for ``rows``. Pixel checks always run once per image; with
    ``find_box``, photos that have no box get one from the detector."""
    checks = read_cache(data_dir)
    images = data_dir / "processed" / "images"
    done = 0
    for r in rows:
        entry = checks.get(r.image_id)
        need_pixels = entry is None
        need_box = find_box is not None and not r.bbox and not (entry or {}).get("detected")
        if not (need_pixels or need_box):
            continue
        entry = dict(entry or {"pixel": [], "box": None, "detected": False})
        path = stored_path(images, r.image_id)
        if not path.exists():
            continue  # deleted after a "not sure" review
        with Image.open(path) as im:
            img = im.convert("RGB")
        if need_pixels:
            entry["pixel"] = pixel_flags(img, policy)
        if need_box and find_box is not None:
            entry["box"] = find_box(img)
            entry["detected"] = True
        checks[r.image_id] = entry
        done += 1
        if done % 500 == 0:
            write_cache(data_dir, checks)
            log.info("checked %d photos", done)
    write_cache(data_dir, checks)
    return checks


def flags_for(
    r: ManifestRow, entry: dict[str, Any] | None, policy: QualityPolicy = DEFAULT_QUALITY
) -> list[str]:
    flags = list((entry or {}).get("pixel", []))
    if r.bbox:
        flags += box_flags(r.bbox, policy)
    elif entry and entry.get("detected"):
        flags += box_flags(entry.get("box"), policy)
    return flags


def apply_checks(
    rows: list[ManifestRow], data_dir: Path, policy: QualityPolicy = DEFAULT_QUALITY
) -> list[ManifestRow]:
    """Add the quality flags to each row (computing pixel checks for new photos)."""
    checks = check_photos(rows, data_dir, policy=policy)
    out = []
    for r in rows:
        new = set(flags_for(r, checks.get(r.image_id), policy))
        kept = {f for f in r.flags if f not in QUALITY_FLAGS}
        out.append(replace(r, flags=sorted(kept | new)))
    return out


def is_weak(r: ManifestRow) -> bool:
    """A flagged photo I haven't vouched for in review."""
    return bool(QUALITY_FLAGS & set(r.flags)) and "reviewed" not in r.flags


def rfdetr_box_finder(size: str = "base") -> BoxFinder:
    """The largest confident cat or dog box from RF-DETR, normalised, or ``None``."""
    from petlens.vision.detect import DetectorConfig, load_rfdetr

    detect = load_rfdetr(DetectorConfig(size=size))

    def find(img: Image.Image) -> list[float] | None:
        animals = [d for d in detect(img) if d.label in ("cat", "dog")]
        if not animals:
            return None
        best = max(animals, key=lambda d: d.area)
        return best.normalised(*img.size)

    return find


def main(argv: list[str] | None = None) -> None:
    from petlens.config import get_settings
    from petlens.data.dataset import load_source_manifests

    parser = argparse.ArgumentParser(prog="python -m petlens.data.quality")
    parser.add_argument("--no-detector", action="store_true", help="pixel checks only")
    parser.add_argument("--detector-size", default="base")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data_dir = get_settings().data_dir
    rows = load_source_manifests(data_dir)
    finder = None if args.no_detector else rfdetr_box_finder(args.detector_size)
    checks = check_photos(rows, data_dir, find_box=finder)
    counts: Counter[str] = Counter()
    for r in rows:
        counts.update(flags_for(r, checks.get(r.image_id)))
    print(f"checked {len(rows)} photos")
    for flag in sorted(QUALITY_FLAGS):
        print(f"  {flag}: {counts[flag]}")
    print("Run `make data-build` to apply the flags, then `make review-load`.")


if __name__ == "__main__":
    main()
