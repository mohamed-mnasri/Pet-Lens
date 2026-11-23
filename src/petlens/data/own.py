"""My own photos of one breed, added a batch at a time from the page's "Add training
photos" tab (``POST /training-photos``). ``make own-photos`` lists what was loaded.

Every photo in a batch gets the breed I picked. Each one is checked first and left
out, with the reason, if it would teach the model the wrong thing:

- it doesn't open, or it's too small or nearly black
- no cat or dog is found, or more than one animal is (one breed per photo keeps the
  label right)
- the animal is the wrong species for the breed (a dog in a Ragdoll batch)
- it's black and white, blurry, the animal is cut off or tiny (``petlens.data.quality``)

People's faces are blurred before the photo is stored. The kept copy is the same
compressed WebP as every dataset image, in ``D:\\petlens-data\\processed\\images``,
listed in ``processed/sources/own/manifest.parquet`` with ``label_origin = me``. The
same photo sent twice is kept once. ``make data-build`` then splits the photos one by
one like Commons photos (most to training, a few held out for testing).
"""

from __future__ import annotations

import argparse
import io
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from petlens.data.catalog import BreedCatalog
from petlens.data.imaging import (
    ImageRejectedError,
    decode,
    quality_flags,
    sha256_bytes,
    store_image,
)
from petlens.data.manifest import ManifestRow, read_manifest, write_manifest
from petlens.data.quality import box_flags, pixel_flags
from petlens.vision.detect import Detection, Detector
from petlens.vision.privacy import blur_people

SOURCE_KEY = "own"
REASONS = {
    "too_small": "too small (short side under 224 px)",
    "near_black": "almost black",
    "grayscale": "black and white",
    "blurry": "blurry",
    "cut_off": "only part of the animal is in the photo",
    "small_animal": "the animal is too small in the photo",
}


@dataclass(frozen=True)
class AddResult:
    status: Literal["added", "already", "left_out"]
    reason: str | None = None
    image_id: str | None = None


_lock = threading.Lock()  # one manifest writer at a time


def manifest_path(data_dir: Path, source: str = SOURCE_KEY) -> Path:
    return data_dir / "processed" / "sources" / source / "manifest.parquet"


def _problem(
    img: Image.Image, found: list[Detection], species: str
) -> tuple[str | None, list[float]]:
    """Why the photo can't be used (or ``None``), and the animal's normalised box."""
    flags = quality_flags(img)
    if flags:
        return REASONS[flags[0]], []
    animals = [d for d in found if d.label in ("cat", "dog")]
    if not animals:
        return "no cat or dog found", []
    if len(animals) > 1:
        return f"{len(animals)} animals in one photo (use photos with one)", []
    animal = animals[0]
    if species != animal.label and species not in animal.alternatives:
        return f"looks like a {animal.label}, not a {species}", []
    box = animal.normalised(*img.size)
    weak = [*pixel_flags(img), *box_flags(box)]
    if weak:
        return REASONS[weak[0]], []
    return None, box


def add_photo(
    data: bytes,
    breed: str,
    data_dir: Path,
    breeds: BreedCatalog,
    detect: Detector,
    licence: str = "own-photo",
    attribution: str = "Mohamed",
    source: str = SOURCE_KEY,
    label_origin: str = "me",
    ref: str | None = None,
    author: str | None = None,
    extra_flags: tuple[str, ...] = (),
) -> AddResult:
    """Check one photo and, if it's usable, store it and list it in ``source``'s
    manifest. ``ref`` names it within the source (default: the original's hash)."""
    entry = breeds.get(breed)
    if entry.role != "train":
        raise ValueError(f"{entry.name} isn't a breed the model trains on")
    ref = ref or sha256_bytes(data)  # so the same photo is kept once
    try:
        img = decode(data)
    except ImageRejectedError:
        return AddResult("left_out", "can't open it as a photo")
    found = detect(img)
    problem, box = _problem(img, found, entry.species)
    if problem:
        return AddResult("left_out", problem)
    with _lock:
        path = manifest_path(data_dir, source)
        rows = read_manifest(path) if path.exists() else []
        same = next((r for r in rows if r.source_ref == ref), None)
        if same is not None:
            return AddResult("already", None, same.image_id)
        buf = io.BytesIO()
        blur_people(img, found).save(buf, "PNG")
        stored = store_image(buf.getvalue(), data_dir / "processed" / "images")
        rows.append(
            ManifestRow(
                image_id=stored.image_id,
                source=source,
                source_ref=ref,
                licence=licence,
                attribution=attribution,
                species=entry.species,
                breed=breed,
                label_origin=label_origin,
                width=stored.original_width,
                height=stored.original_height,
                author=author,
                bbox=box,
                flags=sorted({*stored.flags, *extra_flags}),
            )
        )
        write_manifest(rows, path)
    return AddResult("added", None, stored.image_id)


def known_refs(data_dir: Path, source: str) -> set[str]:
    path = manifest_path(data_dir, source)
    return {r.source_ref for r in read_manifest(path)} if path.exists() else set()


def counts(data_dir: Path, source: str = SOURCE_KEY) -> dict[str, int]:
    """Photos loaded so far, per breed."""
    path = manifest_path(data_dir, source)
    rows = read_manifest(path) if path.exists() else []
    return dict(sorted(Counter(r.breed for r in rows).items()))


def main(argv: list[str] | None = None) -> None:
    from petlens.config import get_settings

    argparse.ArgumentParser(
        prog="python -m petlens.data.own", description="list my own training photos per breed"
    ).parse_args(argv)
    found = counts(get_settings().data_dir)
    if not found:
        print("No training photos loaded yet. Use the page's 'Add training photos' tab.")
        return
    width = max(map(len, found))
    for breed, n in found.items():
        print(f"{breed:<{width}}  {n}")
    print(f"{'total':<{width}}  {sum(found.values())}")
    print("Next: make data-build, make embed, make train, then make compare.")


if __name__ == "__main__":
    main()
