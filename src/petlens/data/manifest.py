"""Manifest schema shared by every source, plus Parquet read/write.

Columns follow DATASET.md. Columns filled by later stages (``phash``, ``group_id``,
``split``, ``added_in``) stay null until those stages run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("source", pa.string()),
        ("source_ref", pa.string()),
        ("licence", pa.string()),
        ("attribution", pa.string()),
        ("species", pa.string()),
        ("breed", pa.string()),
        ("label_origin", pa.string()),
        ("group_id", pa.string()),
        ("cat_id", pa.string()),
        ("video_id", pa.string()),
        ("frame_idx", pa.int32()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("phash", pa.string()),
        ("bbox", pa.list_(pa.float32(), 4)),
        ("flags", pa.list_(pa.string())),
        ("source_split", pa.string()),
        ("split", pa.string()),
        ("added_in", pa.string()),
    ]
)


@dataclass
class ManifestRow:
    image_id: str
    source: str
    source_ref: str
    licence: str
    attribution: str
    species: str
    breed: str
    label_origin: str
    width: int
    height: int
    group_id: str | None = None
    cat_id: str | None = None
    video_id: str | None = None
    frame_idx: int | None = None
    phash: str | None = None
    bbox: list[float] | None = None
    flags: list[str] = field(default_factory=list)
    source_split: str | None = None
    split: str | None = None
    added_in: str | None = None


def to_table(rows: list[ManifestRow]) -> pa.Table:
    rows = sorted(rows, key=lambda r: (r.source, r.source_ref))
    return pa.Table.from_pylist([asdict(r) for r in rows], schema=SCHEMA)


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(to_table(rows), tmp)
    tmp.replace(path)


def read_manifest(path: Path) -> list[ManifestRow]:
    table = pq.read_table(path, schema=SCHEMA)
    return [ManifestRow(**rec) for rec in table.to_pylist()]
