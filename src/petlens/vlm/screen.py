"""The VLM image screener (Step 13): a local vision model tags dataset photos.

``qwen3-vl:4b`` through Ollama (local only) looks at each photo once per prompt version
and answers in JSON, validated with Pydantic: what it is (a real photo, a drawing, a
toy, other), whether the whole animal is visible, whether it is a kitten or puppy,
whether it is blurry, and whether there is more than one animal. An answer that isn't
valid JSON gets one retry, then is stored as ``parse_failed`` (and counted).

The VLM never changes a label. Its tags only keep weak web photos out of training:
``make vlm-export`` writes them to ``data/vlm/tags.parquet`` (committed), ``make
data-build`` turns them into manifest flags, and Commons and Openverse photos flagged as
not a real photo, only part of the animal, or several animals are not trained on unless I
reviewed them. Every tag also becomes a failure slice in the dbt marts. It runs as a
worker job, never next to the live camera, because of the 6 GB of VRAM.
"""

from __future__ import annotations

import base64
import io
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from petlens.data.imaging import stored_path
from petlens.data.manifest import ManifestRow
from petlens.data.vlm_flags import SCREENED_SOURCES, TAGS_PATH
from petlens.db import models as m
from petlens.vision.live import live_active

PROMPT_VERSION = "screen-v1"
DEFAULT_MODEL = "qwen3-vl:4b"
DEFAULT_URL = "http://127.0.0.1:11434"
LONG_EDGE = 768

PROMPT = """You check photos for a cat and dog breed dataset. Look at the image and answer
only with JSON matching the schema:
- "kind": "photo" for a real photograph of a real animal, "drawing" for a drawing,
  painting, cartoon or 3D render, "toy" for a toy, figurine or stuffed animal, "other"
  for anything else (no animal, text, a screenshot).
- "whole_animal": true if the main animal's head and most of its body are in the
  frame; false if only part of it shows (just paws, a nose, a close-up of the face).
- "kitten": true if the main animal is a kitten or a puppy.
- "blurry": true if the animal is out of focus or motion-blurred.
- "several_animals": true if more than one cat or dog is clearly visible.
"""


class VlmTags(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["photo", "drawing", "toy", "other"]
    whole_animal: bool
    kitten: bool
    blurry: bool
    several_animals: bool


SCHEMA = VlmTags.model_json_schema()


class Vlm(Protocol):
    model: str

    def digest(self) -> str: ...

    def ask(self, prompt: str, image_jpeg: bytes, schema: dict[str, Any]) -> str: ...


class OllamaVlm:
    """Ollama's chat API with structured output (temperature 0)."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        http: httpx.Client | None = None,
        timeout: float = 180,
    ) -> None:
        self.model = model
        self.http = http or httpx.Client(base_url=base_url, timeout=timeout)

    def digest(self) -> str:
        tags = self.http.get("/api/tags")
        tags.raise_for_status()
        for entry in tags.json().get("models", []):
            if entry.get("name") == self.model or entry.get("model") == self.model:
                return str(entry.get("digest", ""))
        raise RuntimeError(f"{self.model} isn't pulled yet; run `make vlm-pull`")

    def ask(self, prompt: str, image_jpeg: bytes, schema: dict[str, Any]) -> str:
        r = self.http.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64.b64encode(image_jpeg).decode()],
                    }
                ],
                "format": schema,
                "stream": False,
                "options": {"temperature": 0},
            },
        )
        r.raise_for_status()
        return str(r.json()["message"]["content"])


def to_jpeg(path: Path, long_edge: int = LONG_EDGE) -> bytes:
    with Image.open(path) as im:
        img = im.convert("RGB")
    img.thumbnail((long_edge, long_edge), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


@dataclass
class Answer:
    tags: VlmTags | None
    raw: str
    attempts: int
    seconds: float


def parse(raw: str) -> VlmTags | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return VlmTags.model_validate(json.loads(text))
    except (ValueError, ValidationError):
        return None


def screen_one(vlm: Vlm, image_jpeg: bytes) -> Answer:
    """Ask once; if the answer isn't valid, ask once more; then give up."""
    started = time.monotonic()
    raw = ""
    for attempt in (1, 2):
        raw = vlm.ask(PROMPT, image_jpeg, SCHEMA)
        tags = parse(raw)
        if tags is not None:
            return Answer(tags, raw, attempt, time.monotonic() - started)
    return Answer(None, raw, 2, time.monotonic() - started)


def candidates(
    rows: Iterable[ManifestRow], reviewed: set[str], done: set[str]
) -> list[ManifestRow]:
    """Photos to tag, most useful first: ones I reviewed (to measure the VLM against my
    answers), then Commons and Openverse training candidates, then their held-out photos."""
    todo = [
        r
        for r in rows
        if r.image_id not in done
        and r.split != "excluded"
        and (r.source in SCREENED_SOURCES or r.image_id in reviewed)
    ]

    def order(r: ManifestRow) -> tuple[int, str]:
        if r.image_id in reviewed:
            return 0, r.image_id
        return (1 if r.split in ("train", "val", "calibration") else 2), r.image_id

    return sorted(todo, key=order)


def done_ids(session: Session, model_tag: str, prompt_version: str = PROMPT_VERSION) -> set[str]:
    q = select(m.VlmTag.image_id).where(
        m.VlmTag.model_tag == model_tag, m.VlmTag.prompt_version == prompt_version
    )
    return set(session.scalars(q))


def reviewed_ids(session: Session) -> set[str]:
    q = select(m.LabelEvent.image_id).where(
        m.LabelEvent.image_id.is_not(None), m.LabelEvent.origin == "me"
    )
    return {i for i in session.scalars(q) if i}


def run(
    session: Session,
    rows: list[ManifestRow],
    data_dir: Path,
    vlm: Vlm,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Tag up to ``limit`` photos not tagged yet. Each answer is committed on its own, so
    an interrupted run picks up where it stopped."""
    digest = vlm.digest()
    todo = candidates(rows, reviewed_ids(session), done_ids(session, vlm.model))
    todo = todo[:limit] if limit else todo
    images = data_dir / "processed" / "images"
    counts = {"ok": 0, "parse_failed": 0, "missing": 0}
    for i, r in enumerate(todo, 1):
        while live_active(data_dir):  # the camera page has the GPU: wait for it to close
            time.sleep(5)
        path = stored_path(images, r.image_id)
        if not path.exists():
            counts["missing"] += 1
            continue
        answer = screen_one(vlm, to_jpeg(path))
        t = answer.tags
        session.add(
            m.VlmTag(
                image_id=r.image_id,
                prompt_version=PROMPT_VERSION,
                model_tag=vlm.model,
                model_digest=digest,
                status="ok" if t else "parse_failed",
                kind=t.kind if t else None,
                whole_animal=t.whole_animal if t else None,
                kitten=t.kitten if t else None,
                blurry=t.blurry if t else None,
                several_animals=t.several_animals if t else None,
                attempts=answer.attempts,
                seconds=answer.seconds,
                raw=answer.raw[:4000],
            )
        )
        session.commit()
        counts["ok" if t else "parse_failed"] += 1
        if progress:
            progress(i, len(todo))
    return counts


# --- tags into the dataset ----------------------------------------------------------


def export(session: Session, model_tag: str = DEFAULT_MODEL, path: Path = TAGS_PATH) -> int:
    q = select(m.VlmTag).where(
        m.VlmTag.model_tag == model_tag,
        m.VlmTag.prompt_version == PROMPT_VERSION,
        m.VlmTag.status == "ok",
    )
    tags = sorted(session.scalars(q), key=lambda t: t.image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "image_id": pa.array([t.image_id for t in tags], pa.string()),
            "kind": pa.array([t.kind for t in tags], pa.string()),
            "whole_animal": pa.array([t.whole_animal for t in tags], pa.bool_()),
            "kitten": pa.array([t.kitten for t in tags], pa.bool_()),
            "blurry": pa.array([t.blurry for t in tags], pa.bool_()),
            "several_animals": pa.array([t.several_animals for t in tags], pa.bool_()),
        }
    )
    pq.write_table(table, path)
    return len(tags)
