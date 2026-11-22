"""Sample one frame per second from a short clip (at most 10 frames).

Uses PyAV, which ships its own FFmpeg. Duration limits are enforced at the API in
Step 6; this module only reads frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

MAX_FRAMES = 10


@dataclass(frozen=True)
class Frame:
    index: int
    seconds: float
    image: Image.Image


def sample_frames(
    path: Path, every_seconds: float = 1.0, max_frames: int = MAX_FRAMES
) -> list[Frame]:
    """Decode frames at roughly ``0, every_seconds, 2 * every_seconds, ...``."""
    import av

    frames: list[Frame] = []
    container: Any = av.open(str(path))
    try:
        stream = container.streams.video[0]
        rotation = _rotation(stream)
        next_t = 0.0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            if t + 1e-6 < next_t:
                continue
            img = frame.to_image()
            if rotation:
                img = img.rotate(-rotation, expand=True)
            frames.append(Frame(len(frames), t, ImageOps.exif_transpose(img)))
            next_t = (len(frames)) * every_seconds
            if len(frames) >= max_frames:
                break
    finally:
        container.close()
    return frames


def _rotation(stream: Any) -> int:
    """Phone clips often carry a display rotation in metadata."""
    for source in (
        getattr(stream, "metadata", {}) or {},
        getattr(stream.codec_context, "metadata", {}) or {},
    ):
        value = source.get("rotate")
        if value:
            try:
                return int(value) % 360
            except ValueError:
                return 0
    return 0
