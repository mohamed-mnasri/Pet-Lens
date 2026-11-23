"""VLM screener tags as manifest flags (Step 13), kept light so the data pipeline and
training don't import the screener itself.

``data/vlm/tags.parquet`` (``make vlm-export``) holds one row per tagged photo. ``make
data-build`` adds ``vlm_not_photo`` (a drawing, toy or no animal), ``vlm_partial`` (only
part of the animal), ``vlm_several`` (more than one animal), ``vlm_kitten`` and
``vlm_blurry``. The first three keep Commons and Openverse photos out of training unless
I reviewed them; all five show up as failure slices.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from petlens.data.catalog import REPO_DATA_DIR
from petlens.data.manifest import ManifestRow

TAGS_PATH = REPO_DATA_DIR / "vlm" / "tags.parquet"
SCREENED_SOURCES = frozenset({"commons", "openverse"})
BLOCKING = frozenset({"vlm_not_photo", "vlm_partial", "vlm_several"})
VLM_FLAGS = BLOCKING | {"vlm_kitten", "vlm_blurry"}


def read_tags(path: Path = TAGS_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {d["image_id"]: d for d in pq.read_table(path).to_pylist()}


def flags_from(tag: dict[str, Any]) -> set[str]:
    flags = set()
    if tag["kind"] != "photo":
        flags.add("vlm_not_photo")
    if not tag["whole_animal"]:
        flags.add("vlm_partial")
    if tag["several_animals"]:
        flags.add("vlm_several")
    if tag["kitten"]:
        flags.add("vlm_kitten")
    if tag["blurry"]:
        flags.add("vlm_blurry")
    return flags


def apply_tags(rows: list[ManifestRow], tags: dict[str, dict[str, Any]]) -> list[ManifestRow]:
    out = []
    for r in rows:
        kept = {f for f in r.flags if f not in VLM_FLAGS}
        tag = tags.get(r.image_id)
        new = flags_from(tag) if tag else set()
        out.append(replace(r, flags=sorted(kept | new)) if new or kept != set(r.flags) else r)
    return out


def blocked(r: ManifestRow) -> bool:
    """A web photo the VLM says isn't a usable photo of one whole animal, and that I
    haven't reviewed. My own, Oxford and reviewed photos are never blocked."""
    return r.source in SCREENED_SOURCES and r.label_origin != "me" and bool(BLOCKING & set(r.flags))
