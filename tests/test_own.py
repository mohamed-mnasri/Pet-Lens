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


def test_breed_folders_are_matched_and_imported(tmp_path: Path) -> None:
    from petlens.data import folders

    assert folders.match_breed("Maine Coon cats", BREEDS) == "maine_coon"
    assert folders.match_breed("scottish-fold", BREEDS) == "scottish_fold"
    assert folders.match_breed("Himalayans", BREEDS) == "himalayan"
    assert folders.match_breed("Unicorn", BREEDS) is None
    root = tmp_path / "cats"
    (root / "Ragdoll").mkdir(parents=True)
    (root / "Unicorn").mkdir()
    (root / "Ragdoll" / "a.jpg").write_bytes(_png(_photo(7)))
    (root / "Ragdoll" / "b.png").write_bytes(_png(_photo(8).convert("L").convert("RGB")))
    data = tmp_path / "data"
    report = folders.import_folders(root, "Pexels licence", data, BREEDS, _detector(CAT))
    assert report.added["ragdoll"] == 1 and report.left_out["ragdoll"] == 1
    assert report.unmatched == ["Unicorn"] and "black and white: 1" in report.render()
    again = folders.import_folders(root, "Pexels licence", data, BREEDS, _detector(CAT))
    assert again.already["ragdoll"] == 1 and not again.added
    (row,) = read_manifest(own.manifest_path(data, "licensed"))
    assert (row.licence, row.source, row.label_origin) == ("Pexels licence", "licensed", "me")


def test_openverse_keeps_licensed_photos_and_stops_at_the_limit(tmp_path: Path) -> None:
    import json

    from petlens.data import openverse

    results = [
        {"id": "w", "url": "u/w", "license": "by", "source": "wikimedia"},
        {"id": "nc", "url": "u/nc", "license": "by-nc", "source": "flickr"},
        *[
            {
                "id": f"f{i}",
                "url": f"u/{i}",
                "license": "by",
                "license_version": "2.0",
                "source": "flickr",
                "creator": "Ann",
                "title": "Cat",
                "foreign_landing_url": f"https://flickr/{i}",
            }
            for i in range(4)
        ],
    ]

    def get(url: str) -> bytes:
        if url.startswith(openverse.API_URL):
            page = int(dict(p.split("=") for p in url.split("?")[1].split("&"))["page"])
            return json.dumps({"results": results if page == 1 else []}).encode()
        return _png(_photo(int(url.split("/")[1][-1]) + 20 if url[-1].isdigit() else 0))

    got = openverse.fetch_breed(
        "birman", BREEDS, tmp_path, _detector(CAT), per_breed=3, get=get, sleep=lambda s: None
    )
    assert got["added"] == 3 and got["skipped (Wikimedia or licence)"] == 2
    rows = read_manifest(own.manifest_path(tmp_path, "openverse"))
    assert {r.licence for r in rows} == {"CC BY 2.0"}
    assert all(r.label_origin == "openverse_search" and r.author == "Ann" for r in rows)
    assert "https://flickr/0" in rows[0].attribution
    again = openverse.fetch_breed(
        "birman", BREEDS, tmp_path, _detector(CAT), per_breed=3, get=get, sleep=lambda s: None
    )
    assert not again  # already has 3
