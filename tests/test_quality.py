"""Weak-photo checks: black and white, blurry, cut off, tiny, no animal."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from petlens.data import quality
from petlens.data.catalog import load_breeds
from petlens.data.imaging import store_image
from petlens.models.train import in_training_pool
from petlens.review import cards
from petlens.ui import view
from test_review import RESULTS, ROWS, _row


def _photo(seed: int = 0, size: tuple[int, int] = (640, 480)) -> Image.Image:
    """Colourful, detailed noise: sharp and clearly in colour."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    return Image.fromarray(a).resize(size, Image.Resampling.NEAREST)


def _bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_pixel_flags() -> None:
    img = _photo()
    assert quality.pixel_flags(img) == []
    assert quality.pixel_flags(img.convert("L").convert("RGB")) == ["grayscale"]
    assert quality.pixel_flags(img.filter(ImageFilter.GaussianBlur(6))) == ["blurry"]


def test_box_flags() -> None:
    assert quality.box_flags([0.2, 0.2, 0.8, 0.9]) == []
    assert quality.box_flags([0.0, 0.0, 1.0, 1.0]) == ["cut_off"]  # paws or nose only
    assert quality.box_flags([0.0, 0.3, 1.0, 1.0]) == []  # lying across the bottom: fine
    assert quality.box_flags([0.9, 0.9, 0.95, 0.95]) == ["small_animal"]
    assert quality.box_flags(None) == ["no_animal"]


def test_checks_are_cached_and_applied(tmp_path: Path) -> None:
    images = tmp_path / "processed" / "images"
    sharp = store_image(_bytes(_photo(1)), images).image_id
    grey = store_image(_bytes(_photo(2).convert("L").convert("RGB")), images).image_id
    rows = [
        _row(sharp, "bengal", bbox=None),
        _row(grey, "ragdoll", bbox=[0.1, 0.1, 0.9, 0.9], flags=["too_small"]),
    ]
    out = {r.image_id: r for r in quality.apply_checks(rows, tmp_path)}
    assert out[sharp].flags == []  # no box yet, and no detector run: no box flags
    assert out[grey].flags == ["grayscale", "too_small"]
    # The detector pass finds nothing in the first photo; the second already has a box.
    seen: list[str] = []

    def finder(img: Image.Image) -> list[float] | None:
        seen.append("x")
        return None

    quality.check_photos(rows, tmp_path, find_box=finder)
    quality.check_photos(rows, tmp_path, find_box=finder)
    assert seen == ["x"]  # cached after the first run
    out = {r.image_id: r for r in quality.apply_checks(rows, tmp_path)}
    assert out[sharp].flags == ["no_animal"]


def test_weak_photos_stay_out_of_review_and_training() -> None:
    weak = [
        _row(r.image_id, r.breed, flags=["grayscale"]) if r.image_id[0] == "b" else r for r in ROWS
    ]
    made = {cd["image_id"][0] for cd in cards.commons_cards(weak, RESULTS)}
    assert made == {"a", "c"}
    oxford = _row("e" * 64, "bengal", source="oxford", flags=["blurry"])
    assert not in_training_pool(oxford, frozenset())
    vouched = _row("e" * 64, "bengal", source="oxford", flags=["blurry", "reviewed"])
    assert in_training_pool(vouched, frozenset())
    checks = cards.check_cards(
        [_row("f" * 64, "bengal", source="oxford", split="test_in_domain", flags=["cut_off"])],
        load_breeds(),
        n=5,
    )
    assert checks == []


def test_upload_tips() -> None:
    good = _bytes(_photo(3))
    whole = {"species": "cat", "x0": 0.2, "y0": 0.2, "x1": 0.8, "y1": 0.9}
    close = {"species": "cat", "x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
    assert view.photo_tips(good, [whole]) == []
    assert view.photo_tips(good, [close]) == [view.TIPS["cut_off"]]
    assert view.photo_tips(good, [whole, close]) == []  # one cat is framed well
    grey = _bytes(_photo(3).convert("L").convert("RGB"))
    assert view.photo_tips(grey, []) == [view.TIPS["grayscale"]]
    assert "face and the body" in view.PHOTO_GUIDE
