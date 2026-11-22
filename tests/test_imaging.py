import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from petlens.data.imaging import (
    ImageRejectedError,
    StoragePolicy,
    encode_webp,
    store_image,
    stored_path,
)


def _noise_jpeg(w: int, h: int, seed: int = 0, exif: Image.Exif | None = None) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    kwargs = {"exif": exif} if exif is not None else {}
    Image.fromarray(arr).save(buf, format="JPEG", quality=95, **kwargs)
    return buf.getvalue()


def test_store_resizes_to_long_edge_and_names_by_hash(tmp_path: Path) -> None:
    data = _noise_jpeg(2000, 1000)
    s = store_image(data, tmp_path)
    assert s.path == stored_path(tmp_path, s.image_id)
    assert (s.width, s.height) == (1024, 512)
    assert (s.original_width, s.original_height) == (2000, 1000)
    with Image.open(s.path) as img:
        assert img.format == "WEBP"


def test_noise_image_still_fits_under_cap(tmp_path: Path) -> None:
    # Random noise is the worst case for compression.
    s = store_image(_noise_jpeg(1600, 1600, seed=1), tmp_path)
    assert s.nbytes <= 250_000


def test_exif_orientation_applied_and_metadata_dropped(tmp_path: Path) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    exif[0x8825] = {1: "N", 2: (52.0, 22.0, 0.0)}  # a GPS block
    s = store_image(_noise_jpeg(600, 300, exif=exif), tmp_path)
    assert (s.width, s.height) == (300, 600)
    with Image.open(s.path) as img:
        assert not img.getexif()
        assert "exif" not in img.info


def test_rerun_reuses_stored_file(tmp_path: Path) -> None:
    data = _noise_jpeg(500, 400)
    first = store_image(data, tmp_path)
    mtime = first.path.stat().st_mtime_ns
    second = store_image(data, tmp_path)
    assert second == first
    assert first.path.stat().st_mtime_ns == mtime


def test_flags_small_and_near_black(tmp_path: Path) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (2, 2, 2)).save(buf, format="PNG")
    s = store_image(buf.getvalue(), tmp_path)
    assert set(s.flags) == {"too_small", "near_black"}


def test_transparent_png_goes_on_white(tmp_path: Path) -> None:
    buf = io.BytesIO()
    Image.new("RGBA", (400, 400), (0, 0, 0, 0)).save(buf, format="PNG")
    s = store_image(buf.getvalue(), tmp_path)
    with Image.open(s.path) as img:
        assert img.convert("RGB").getpixel((10, 10)) == (255, 255, 255)


def test_garbage_rejected(tmp_path: Path) -> None:
    with pytest.raises(ImageRejectedError):
        store_image(b"not an image at all", tmp_path)


def test_encode_steps_down_until_it_fits() -> None:
    rng = np.random.default_rng(2)
    img = Image.fromarray(rng.integers(0, 256, size=(1024, 1024, 3), dtype=np.uint8))
    policy = StoragePolicy(max_bytes=40_000)
    assert len(encode_webp(img, policy)) <= 40_000
