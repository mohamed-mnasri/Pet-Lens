"""Oxford-IIIT Pet: fetch, extract, ingest and write the source manifest.

Annotation format (``annotations/list.txt`` and the split files): one line per image,
``<name> <class_id> <species> <breed_id>``, where species 1 is cat and 2 is dog. Trimap
pixels are 1 for the animal, 2 for background and 3 for the boundary.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from petlens.data.catalog import BreedCatalog, Source
from petlens.data.download import extract, fetch
from petlens.data.imaging import DEFAULT_POLICY, ImageRejectedError, StoragePolicy, store_image
from petlens.data.manifest import ManifestRow, write_manifest

log = logging.getLogger(__name__)

SOURCE_KEY = "oxford"
_SPECIES = {"1": "cat", "2": "dog"}


@dataclass(frozen=True)
class OxfordEntry:
    name: str
    species: str
    source_split: str


@dataclass
class IngestReport:
    total_listed: int = 0
    kept: int = 0
    rejected: dict[str, str] = field(default_factory=dict)
    exact_duplicates: dict[str, str] = field(default_factory=dict)
    per_breed: dict[str, dict[str, int]] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    stored_bytes: int = 0


def read_split_files(annotations_dir: Path) -> list[OxfordEntry]:
    """Parse ``trainval.txt`` and ``test.txt`` into entries, sorted by name."""
    entries: dict[str, OxfordEntry] = {}
    for split in ("trainval", "test"):
        for line in (annotations_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _class_id, species, _breed_id = line.split()
            if name in entries:
                raise ValueError(f"{name} is in more than one official split")
            entries[name] = OxfordEntry(name, _SPECIES[species], split)
    return [entries[k] for k in sorted(entries)]


def oxford_label(name: str) -> str:
    """``British_Shorthair_12`` -> ``British_Shorthair``."""
    return name.rsplit("_", 1)[0]


def body_bbox(trimap_path: Path) -> list[float] | None:
    """Normalised ``[x0, y0, x1, y1]`` around animal and boundary pixels, or None."""
    with Image.open(trimap_path) as tm:
        arr = np.asarray(tm)
    mask = (arr == 1) | (arr == 3)
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return [
        round(float(xs.min()) / w, 4),
        round(float(ys.min()) / h, 4),
        round(float(xs.max() + 1) / w, 4),
        round(float(ys.max() + 1) / h, 4),
    ]


@dataclass(frozen=True)
class _Job:
    entry: OxfordEntry
    image_path: Path
    trimap_path: Path
    images_dir: Path
    policy: StoragePolicy


@dataclass(frozen=True)
class _Result:
    entry: OxfordEntry
    image_id: str | None = None
    width: int = 0
    height: int = 0
    nbytes: int = 0
    bbox: list[float] | None = None
    flags: tuple[str, ...] = ()
    error: str | None = None


def _ingest_one(job: _Job) -> _Result:
    if not job.image_path.exists():
        return _Result(job.entry, error="image file missing")
    try:
        stored = store_image(job.image_path.read_bytes(), job.images_dir, job.policy)
    except ImageRejectedError as exc:
        return _Result(job.entry, error=str(exc))
    flags = list(stored.flags)
    bbox = None
    if not job.trimap_path.exists():
        flags.append("no_trimap")
    else:
        with Image.open(job.trimap_path) as tm:
            tm_size = tm.size
        if tm_size != (stored.original_width, stored.original_height):
            flags.append("trimap_size_mismatch")
        else:
            bbox = body_bbox(job.trimap_path)
            if bbox is None:
                flags.append("empty_trimap")
    return _Result(
        job.entry,
        image_id=stored.image_id,
        width=stored.width,
        height=stored.height,
        nbytes=stored.nbytes,
        bbox=bbox,
        flags=tuple(flags),
    )


def ingest(
    extracted_dir: Path,
    images_dir: Path,
    breeds: BreedCatalog,
    source: Source,
    policy: StoragePolicy = DEFAULT_POLICY,
    workers: int = 1,
) -> tuple[list[ManifestRow], IngestReport]:
    """Store every listed Oxford image and build its manifest rows."""
    annotations = extracted_dir / "annotations"
    entries = read_split_files(annotations)
    label_map = breeds.by_oxford_label()
    jobs = []
    for e in entries:
        label = oxford_label(e.name)
        breed = label_map.get(label)
        if breed is None:
            raise ValueError(f"Oxford label {label!r} is not in breeds.yaml")
        if breed.species != e.species:
            raise ValueError(f"{e.name}: species {e.species} disagrees with breeds.yaml")
        jobs.append(
            _Job(
                e,
                extracted_dir / "images" / f"{e.name}.jpg",
                annotations / "trimaps" / f"{e.name}.png",
                images_dir,
                policy,
            )
        )

    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            results = list(pool.map(_ingest_one, jobs, chunksize=32))
    else:
        results = [_ingest_one(j) for j in jobs]

    report = IngestReport(total_listed=len(entries))
    rows: list[ManifestRow] = []
    seen: dict[str, str] = {}
    flag_counts: Counter[str] = Counter()
    for r in results:  # results keep the sorted entry order, so "first seen" is stable
        if r.error is not None or r.image_id is None:
            report.rejected[r.entry.name] = r.error or "unknown error"
            continue
        if r.image_id in seen:
            report.exact_duplicates[r.entry.name] = seen[r.image_id]
            continue
        seen[r.image_id] = r.entry.name
        breed_id = label_map[oxford_label(r.entry.name)].id
        rows.append(
            ManifestRow(
                image_id=r.image_id,
                source=SOURCE_KEY,
                source_ref=f"images/{r.entry.name}.jpg",
                licence=source.licence,
                attribution=source.attribution,
                species=r.entry.species,
                breed=breed_id,
                label_origin=SOURCE_KEY,
                width=r.width,
                height=r.height,
                bbox=r.bbox,
                flags=list(r.flags),
                source_split=r.entry.source_split,
            )
        )
        counts = report.per_breed.setdefault(breed_id, {"trainval": 0, "test": 0})
        counts[r.entry.source_split] += 1
        flag_counts.update(r.flags)
        report.stored_bytes += r.nbytes
    report.kept = len(rows)
    report.flags = dict(sorted(flag_counts.items()))
    report.per_breed = dict(sorted(report.per_breed.items()))
    return rows, report


def render_report(report: IngestReport) -> str:
    lines = [
        "# Oxford-IIIT Pet ingest report",
        "",
        f"- Listed in the official splits: {report.total_listed}",
        f"- Kept: {report.kept}",
        f"- Rejected (could not be used): {len(report.rejected)}",
        f"- Exact duplicates dropped: {len(report.exact_duplicates)}",
        f"- Stored size: {report.stored_bytes / 1e6:.1f} MB "
        f"({report.stored_bytes / max(report.kept, 1) / 1e3:.0f} KB per image on average)",
        "",
        "## Images per breed",
        "",
        "| Breed | Trainval | Test |",
        "| --- | ---: | ---: |",
    ]
    for breed, c in report.per_breed.items():
        lines.append(f"| {breed} | {c['trainval']} | {c['test']} |")
    lines += ["", "## Flags", ""]
    lines += [f"- {k}: {v}" for k, v in report.flags.items()] or ["- none"]
    if report.rejected:
        lines += ["", "## Rejected", ""]
        lines += [f"- {k}: {v}" for k, v in sorted(report.rejected.items())]
    if report.exact_duplicates:
        lines += ["", "## Exact duplicates (dropped, kept the first)", ""]
        lines += [f"- {k} = {v}" for k, v in sorted(report.exact_duplicates.items())]
    return "\n".join(lines) + "\n"


def build(
    data_dir: Path,
    breeds: BreedCatalog,
    source: Source,
    policy: StoragePolicy = DEFAULT_POLICY,
    workers: int = 1,
) -> Path:
    """Run fetch, extract and ingest. Returns the manifest path. Safe to rerun."""
    raw = data_dir / "raw" / SOURCE_KEY / source.version
    extracted = raw / "extracted"
    for spec in source.files:
        extract(fetch(spec, raw), extracted)
    rows, report = ingest(
        extracted, data_dir / "processed" / "images", breeds, source, policy, workers
    )
    out_dir = data_dir / "processed" / "sources" / SOURCE_KEY
    manifest = out_dir / "manifest.parquet"
    write_manifest(rows, manifest)
    (out_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    report_json = {k: v for k, v in report.__dict__.items()}
    (out_dir / "report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    log.info("oxford: kept %d of %d listed images", report.kept, report.total_listed)
    return manifest
