"""Turn an incoming image into the one compressed copy PetLens keeps.

The same rules apply to dataset images and, from Step 6, to uploads: apply EXIF
orientation once, drop all metadata (including GPS), resize to a fixed long edge and
re-encode as WebP under a hard byte cap.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

# Pillow's default bomb guard is ~89 MP; the upload limit (Step 6) is 40 MP, so stay above it.
Image.MAX_IMAGE_PIXELS = 100_000_000


@dataclass(frozen=True)
class StoragePolicy:
    long_edge: int = 1024
    quality: int = 80
    min_quality: int = 40
    quality_step: int = 10
    max_bytes: int = 250_000
    min_short_side: int = 224
    near_black_mean: float = 10.0


DEFAULT_POLICY = StoragePolicy()


class ImageRejectedError(ValueError):
    """The file can't be used: it doesn't decode or is otherwise unusable."""


@dataclass(frozen=True)
class StoredImage:
    image_id: str
    path: Path
    width: int
    height: int
    original_width: int
    original_height: int
    nbytes: int
    flags: tuple[str, ...]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stored_path(images_dir: Path, image_id: str) -> Path:
    return images_dir / image_id[:2] / f"{image_id}.webp"


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def decode(data: bytes) -> Image.Image:
    """Decode fully, apply EXIF orientation and convert to RGB."""
    try:
        opened = Image.open(io.BytesIO(data))
        opened.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageRejectedError(f"cannot decode: {exc}") from exc
    return _to_rgb(ImageOps.exif_transpose(opened))


def resize_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    scale = long_edge / max(w, h)
    if scale >= 1:
        return img
    return img.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS
    )


def encode_webp(img: Image.Image, policy: StoragePolicy = DEFAULT_POLICY) -> bytes:
    """Encode as WebP, stepping quality down (then size) until it fits ``max_bytes``."""
    current = img
    while True:
        quality = policy.quality
        while True:
            buf = io.BytesIO()
            # No exif/icc/xmp arguments: the stored copy carries no metadata.
            current.save(buf, format="WEBP", quality=quality, method=4)
            data = buf.getvalue()
            if len(data) <= policy.max_bytes:
                return data
            if quality - policy.quality_step < policy.min_quality:
                break
            quality -= policy.quality_step
        w, h = current.size
        if max(w, h) <= 64:
            return data
        current = current.resize(
            (max(1, round(w * 0.85)), max(1, round(h * 0.85))), Image.Resampling.LANCZOS
        )


def quality_flags(img: Image.Image, policy: StoragePolicy = DEFAULT_POLICY) -> tuple[str, ...]:
    """Cheap checks on the decoded original. Flagged images are kept but excluded later."""
    flags: list[str] = []
    if min(img.size) < policy.min_short_side:
        flags.append("too_small")
    if ImageStat.Stat(img.convert("L")).mean[0] < policy.near_black_mean:
        flags.append("near_black")
    return tuple(flags)


def store_image(
    data: bytes, images_dir: Path, policy: StoragePolicy = DEFAULT_POLICY
) -> StoredImage:
    """Store ``data`` once, named by the SHA-256 of the original bytes.

    Rerunning on the same bytes is cheap: the existing file is reused, not re-encoded.
    """
    image_id = sha256_bytes(data)
    img = decode(data)
    flags = quality_flags(img, policy)
    path = stored_path(images_dir, image_id)
    if path.exists():
        with Image.open(path) as existing:
            width, height = existing.size
        nbytes = path.stat().st_size
    else:
        resized = resize_long_edge(img, policy.long_edge)
        encoded = encode_webp(resized, policy)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(encoded)
        tmp.replace(path)
        with Image.open(path) as written:
            width, height = written.size
        nbytes = len(encoded)
    return StoredImage(
        image_id=image_id,
        path=path,
        width=width,
        height=height,
        original_width=img.width,
        original_height=img.height,
        nbytes=nbytes,
        flags=flags,
    )
