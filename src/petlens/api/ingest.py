"""Upload checks that run before anything is decoded or stored.

File types come from the first bytes (the signature), never from the file name or the
browser's content type. Photo pixel counts come from the image header, and clip
durations from the container metadata (PyAV), so an over-limit file is rejected
without being decoded.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

from petlens.config import Settings

Kind = Literal["photo", "video"]

TRIM_MESSAGE = "Trim the clip to 10 seconds or upload photos instead"


class RejectedUploadError(ValueError):
    """The upload breaks a limit or isn't a supported file. ``status`` is the HTTP code."""

    def __init__(self, reason: str, status: int = 422) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


@dataclass(frozen=True)
class FileType:
    kind: Kind
    mime: str
    suffix: str


def sniff(head: bytes) -> FileType | None:
    """Identify JPEG, PNG, WebP, MP4, MOV and WebM from their first bytes."""
    if head.startswith(b"\xff\xd8\xff"):
        return FileType("photo", "image/jpeg", ".jpg")
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return FileType("photo", "image/png", ".png")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return FileType("photo", "image/webp", ".webp")
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand == b"qt  ":
            return FileType("video", "video/quicktime", ".mov")
        return FileType("video", "video/mp4", ".mp4")
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return FileType("video", "video/webm", ".webm")
    return None


def check_type(data: bytes, expected: Kind) -> FileType:
    ft = sniff(data[:32])
    if ft is None:
        allowed = "JPEG, PNG or WebP" if expected == "photo" else "MP4, MOV or WebM"
        raise RejectedUploadError(f"Unsupported file type: upload a {allowed} file", 415)
    if ft.kind != expected:
        raise RejectedUploadError(f"This is a {ft.kind}; use the {ft.kind} upload instead", 415)
    return ft


def check_size(nbytes: int, kind: Kind, settings: Settings) -> None:
    limit = settings.photo_max_bytes if kind == "photo" else settings.video_max_bytes
    if nbytes > limit:
        raise RejectedUploadError(f"File is over the {limit / 2**20:.0f} MB {kind} limit", 413)


def check_photo_pixels(data: bytes, settings: Settings) -> tuple[int, int]:
    """Read width and height from the header only (no pixel decoding)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RejectedUploadError(f"The photo can't be read: {exc}", 422) from exc
    if w * h > settings.photo_max_pixels:
        raise RejectedUploadError(
            f"Photo is {w * h / 1e6:.0f} megapixels; the limit is "
            f"{settings.photo_max_pixels / 1e6:.0f}",
            413,
        )
    return w, h


def probe_video(path: Path) -> tuple[float, int, int]:
    """Duration in seconds and frame size, from container metadata (nothing is decoded)."""
    import av

    try:
        container: Any = av.open(str(path))
    except Exception as exc:  # PyAV raises several error types for broken files
        raise RejectedUploadError(f"The clip can't be read: {exc}", 422) from exc
    try:
        if not container.streams.video:
            raise RejectedUploadError("The clip has no video track", 422)
        stream = container.streams.video[0]
        seconds: float | None = None
        if container.duration is not None:
            seconds = container.duration / av.time_base
        elif stream.duration is not None and stream.time_base is not None:
            seconds = float(stream.duration * stream.time_base)
        if seconds is None:
            raise RejectedUploadError("The clip's length can't be read", 422)
        return seconds, int(stream.codec_context.width), int(stream.codec_context.height)
    finally:
        container.close()


def check_duration(seconds: float, settings: Settings) -> None:
    if seconds > settings.video_max_seconds + 0.05:  # container rounding
        raise RejectedUploadError(f"Clip is {seconds:.1f} s. {TRIM_MESSAGE}", 413)


_UNSAFE = re.compile(r"[^\w.\- ()]+")


def safe_name(name: str | None) -> str:
    """A display name for the database: no folders, control characters or odd symbols.

    It is never used as a path; stored files are named by content hash.
    """
    base = re.split(r"[\\/]", name or "")[-1]
    base = unicodedata.normalize("NFKC", base)
    base = "".join(ch for ch in base if unicodedata.category(ch)[0] != "C")
    base = _UNSAFE.sub("_", base).strip(" .")
    return base[:255] or "upload"
