"""My own training photos, added one breed batch at a time."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from petlens.data import own
from petlens.data.catalog import load_breeds
from petlens.data.manifest import read_manifest
from petlens.vision.detect import Detection
from test_quality import _photo

BREEDS = load_breeds()
CAT = Detection("cat", 0.9, (100, 80, 500, 440))


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _detector(*found: Detection):  # type: ignore[no-untyped-def]
    return lambda img: list(found)


def test_photos_are_checked_and_kept_once(tmp_path: Path) -> None:
    photo = _png(_photo(5))
    r = own.add_photo(photo, "ragdoll", tmp_path, BREEDS, _detector(CAT))
    assert r.status == "added" and r.image_id
    again = own.add_photo(photo, "ragdoll", tmp_path, BREEDS, _detector(CAT))
    assert (again.status, again.image_id) == ("already", r.image_id)
    own.add_photo(_png(_photo(6)), "birman", tmp_path, BREEDS, _detector(CAT))
    assert own.counts(tmp_path) == {"birman": 1, "ragdoll": 1}
    row = next(x for x in read_manifest(own.manifest_path(tmp_path)) if x.breed == "ragdoll")
    assert (row.source, row.label_origin, row.licence) == ("own", "me", "own-photo")
    assert row.bbox is not None and abs(row.bbox[0] - 100 / 640) < 1e-6


def test_weak_or_ambiguous_photos_are_left_out(tmp_path: Path) -> None:
    def why(img: Image.Image, *found: Detection, breed: str = "ragdoll") -> str:
        r = own.add_photo(_png(img), breed, tmp_path, BREEDS, _detector(*found))
        assert r.status == "left_out" and r.reason
        return r.reason

    assert why(_photo(1)) == "no cat or dog found"
    assert why(_photo(1), CAT, CAT).startswith("2 animals")
    assert why(_photo(1), Detection("dog", 0.9, CAT.box)) == "looks like a dog, not a cat"
    assert why(_photo(1).convert("L").convert("RGB"), CAT) == "black and white"
    assert why(_photo(1), Detection("cat", 0.9, (0, 0, 640, 480))) == (
        "only part of the animal is in the photo"
    )
    assert why(_photo(1, (100, 100)), CAT).startswith("too small")
    assert own.counts(tmp_path) == {}
    r = own.add_photo(b"not a photo", "ragdoll", tmp_path, BREEDS, _detector(CAT))
    assert r.reason == "can't open it as a photo"
