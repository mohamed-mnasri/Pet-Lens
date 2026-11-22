"""Build a dataset version: near-duplicates, groups, splits, manifest and report.

Inputs are the per-source manifests under ``processed/sources/``. The output is a
versioned manifest committed to ``data/manifests/`` plus a copy, a report and image
checksums under ``<DATA_DIR>/manifests/<version>/``.

Split rules (DATASET.md):

- Every image belongs to a group: near-duplicate clusters (pHash) joined with the
  Commons photographer. The photographer is the file's author; the uploader is only
  used when the author is missing or generic, because bulk-transfer accounts (Flickr
  imports) upload many photographers' work. All images of a group land in the same split.
- Oxford keeps its official test split as ``test_in_domain``; a test image that is a
  near-duplicate of an Oxford training image, or a Commons image that duplicates an
  Oxford test image, is excluded (a leak).
- Commons groups are held out by a hash of the group id: some become the unseen-source
  test (domestic cats go to ``test_domestic``), some the real-world calibration slice,
  and the rest join the train/val/calibration pool with Oxford trainval.
- Breeds with role ``unknown`` only ever go to ``test_unknown``. They are grouped by
  near-duplicates only, not by uploader.
- An image that already has a split in the previous version keeps it. New images in a
  group that already has a split follow that group.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from petlens.data.catalog import REPO_DATA_DIR, BreedCatalog
from petlens.data.imaging import stored_path
from petlens.data.manifest import ManifestRow, read_manifest, to_table, write_manifest
from petlens.data.phash import UnionFind, near_duplicate_pairs, phash_file

log = logging.getLogger(__name__)

SPLITS = (
    "train",
    "val",
    "calibration",
    "rw_calibration",
    "test_in_domain",
    "test_unseen_source",
    "test_domestic",
    "test_unknown",
    "excluded",
)
# Splits that are the same held-out partition, named by class. A group may span them.
HELD_OUT = {"test_unseen_source", "test_domestic"}
EXCLUDE_FLAGS = {"too_small", "near_black", "leak_train_test", "dup_of_test", "dup_of_known"}


@dataclass(frozen=True)
class SplitPolicy:
    seed: int = 20261122
    max_phash_distance: int = 6
    held_out_fraction: float = 0.15  # Commons groups -> unseen-source / domestic test
    rw_calibration_fraction: float = 0.10  # Commons groups -> real-world calibration
    val_fraction: float = 0.10  # of the pool
    calibration_fraction: float = 0.10  # of the pool


DEFAULT_SPLIT_POLICY = SplitPolicy()


def _unit(seed: int, salt: str, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _pool_split(group: str, p: SplitPolicy) -> str:
    u = _unit(p.seed, "pool", group)
    if u < p.val_fraction:
        return "val"
    if u < p.val_fraction + p.calibration_fraction:
        return "calibration"
    return "train"


def _commons_split(group: str, is_domestic: bool, p: SplitPolicy) -> str:
    u = _unit(p.seed, "holdout", group)
    if u < p.held_out_fraction:
        return "test_domestic" if is_domestic else "test_unseen_source"
    if u < p.held_out_fraction + p.rw_calibration_fraction:
        return "rw_calibration"
    return _pool_split(group, p)


def partition(split: str) -> str:
    """The partition a split belongs to, for the no-group-crosses-splits check."""
    return "held_out" if split in HELD_OUT else split


# --------------------------------------------------------------------------- stages


def load_source_manifests(data_dir: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for path in sorted((data_dir / "processed" / "sources").glob("*/manifest.parquet")):
        rows.extend(read_manifest(path))
    rows.sort(key=lambda r: (r.source, r.source_ref))
    seen: set[str] = set()
    unique = []
    for r in rows:  # the same bytes from two sources: keep the first
        if r.image_id not in seen:
            seen.add(r.image_id)
            unique.append(r)
    return unique


def add_phashes(rows: list[ManifestRow], data_dir: Path) -> list[ManifestRow]:
    cache_path = data_dir / "processed" / "phash.json"
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    images = data_dir / "processed" / "images"
    out = []
    for r in rows:
        if r.image_id not in cache:
            cache[r.image_id] = phash_file(stored_path(images, r.image_id))
        out.append(replace(r, phash=cache[r.image_id]))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
    return out


_GENERIC_AUTHORS = {"", "unknown", "unknown author", "anonymous", "anonymous author", "author"}


def photographer_key(r: ManifestRow) -> str | None:
    """Who took the photo: the author when it's meaningful, else the uploader."""
    author = " ".join((r.author or "").split()).casefold()
    if author not in _GENERIC_AUTHORS and not author.startswith("unknown"):
        return f"author:{author}"
    if r.uploader:
        return f"uploader:{r.uploader}"
    return None


def assign_groups(
    rows: list[ManifestRow], breeds: BreedCatalog, policy: SplitPolicy
) -> list[ManifestRow]:
    """Mark leaks, then set ``group_id`` from near-duplicate clusters and photographers."""
    role = {b.id: b.role for b in breeds.breeds}
    pairs = near_duplicate_pairs([r.phash or "0" for r in rows], policy.max_phash_distance)

    dup = UnionFind(len(rows))
    for i, j in pairs:
        dup.union(i, j)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        clusters[dup.find(i)].append(i)

    flags = [list(r.flags) for r in rows]
    for members in clusters.values():
        if len(members) < 2:
            continue
        ox_test = [
            i for i in members if rows[i].source == "oxford" and rows[i].source_split == "test"
        ]
        ox_train = [
            i for i in members if rows[i].source == "oxford" and rows[i].source_split != "test"
        ]
        known = [i for i in members if role.get(rows[i].breed) != "unknown"]
        for i in members:
            r = rows[i]
            if i in ox_test and ox_train:
                flags[i].append("leak_train_test")
            elif ox_test and r.source != "oxford":
                flags[i].append("dup_of_test")
            elif role.get(r.breed) == "unknown" and known:
                flags[i].append("dup_of_known")

    # Groups: near-duplicate clusters, joined by photographer for non-unknown Commons rows.
    grp = UnionFind(len(rows))
    for i, j in pairs:
        grp.union(i, j)
    by_photographer: dict[str, int] = {}
    for i, r in enumerate(rows):
        key = photographer_key(r)
        if key and role.get(r.breed) != "unknown":
            first = by_photographer.setdefault(key, i)
            grp.union(first, i)
    members_of: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        members_of[grp.find(i)].append(i)
    group_name: dict[int, str] = {}
    for root, members in members_of.items():
        group_name[root] = "g-" + min(rows[i].image_id for i in members)[:16]

    return [
        replace(r, flags=sorted(set(flags[i])), group_id=group_name[grp.find(i)])
        for i, r in enumerate(rows)
    ]


def assign_splits(
    rows: list[ManifestRow],
    breeds: BreedCatalog,
    previous: dict[str, ManifestRow],
    policy: SplitPolicy,
) -> list[ManifestRow]:
    role = {b.id: b.role for b in breeds.breeds}
    groups: dict[str, list[ManifestRow]] = defaultdict(list)
    for r in rows:
        groups[r.group_id or r.image_id].append(r)

    out: list[ManifestRow] = []
    for gid, members in groups.items():
        # A split already given to any member of this group (from the last version).
        inherited = {
            partition(previous[r.image_id].split or "")
            for r in members
            if r.image_id in previous
            and previous[r.image_id].split not in (None, "excluded")
            and role.get(r.breed) != "unknown"
        }
        has_ox_test = any(r.source == "oxford" and r.source_split == "test" for r in members)
        has_ox_train = any(r.source == "oxford" and r.source_split != "test" for r in members)
        for r in members:
            if set(r.flags) & EXCLUDE_FLAGS:
                split = "excluded"
            elif r.image_id in previous and previous[r.image_id].split not in (None, "excluded"):
                split = previous[r.image_id].split or "excluded"
            elif role.get(r.breed) == "unknown":
                split = "test_unknown"
            elif r.source == "oxford" and r.source_split == "test":
                split = "test_in_domain"
            elif has_ox_test and not has_ox_train:
                split = "excluded"  # shares a group with the in-domain test; keep it out
            elif len(inherited) == 1:
                part = next(iter(inherited))
                if part == "held_out":
                    split = "test_domestic" if r.breed == "domestic_cat" else "test_unseen_source"
                else:
                    split = part
            elif has_ox_train or r.source == "oxford":
                split = _pool_split(gid, policy)
            else:
                split = _commons_split(gid, r.breed == "domestic_cat", policy)
            out.append(replace(r, split=split))
    return out


def content_hash(rows: list[ManifestRow]) -> str:
    """Hash of everything except ``added_in``, so an unchanged rebuild hashes the same."""
    table = to_table([replace(r, added_in=None) for r in rows])
    h = hashlib.sha256()
    for record in table.to_pylist():
        h.update(json.dumps(record, sort_keys=True, default=str).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- report


def render_report(version: str, rows: list[ManifestRow], breeds: BreedCatalog) -> str:
    names = {b.id: b.name for b in breeds.breeds}
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    flags: Counter[str] = Counter()
    licences: Counter[str] = Counter()
    for r in rows:
        by_label[r.breed][r.split or "?"] += 1
        by_source[r.source][r.split or "?"] += 1
        flags.update(r.flags)
        licences[r.licence] += 1
    used = [s for s in SPLITS if any(c[s] for c in by_label.values())]
    group_sizes = Counter(r.group_id for r in rows if r.split != "excluded")
    lines = [
        f"# Dataset {version}",
        "",
        f"- Images: {len(rows)} ({sum(1 for r in rows if r.split != 'excluded')} usable)",
        f"- Groups: {len(group_sizes)} (largest: {max(group_sizes.values(), default=0)} images)",
        f"- Sources: {', '.join(sorted(by_source))}",
        "",
        "## Images per label and split",
        "",
        "| Label | " + " | ".join(used) + " | Total |",
        "| --- | " + " | ".join("---:" for _ in used) + " | ---: |",
    ]
    for label in sorted(by_label, key=lambda b: names.get(b, b)):
        c = by_label[label]
        cells = " | ".join(str(c[s]) for s in used)
        lines.append(f"| {names.get(label, label)} | {cells} | {sum(c.values())} |")
    lines += ["", "## Images per source and split", ""]
    lines += [
        "| Source | " + " | ".join(used) + " |",
        "| --- | " + " | ".join("---:" for _ in used) + " |",
    ]
    for source, c in sorted(by_source.items()):
        lines.append(f"| {source} | " + " | ".join(str(c[s]) for s in used) + " |")
    lines += ["", "## Flags", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(flags.items())] or ["- none"]
    lines += ["", "## Licences", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(licences.items())]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- build


@dataclass(frozen=True)
class BuildResult:
    version: str
    changed: bool
    manifest: Path


def build(
    data_dir: Path,
    breeds: BreedCatalog,
    repo_data_dir: Path = REPO_DATA_DIR,
    policy: SplitPolicy = DEFAULT_SPLIT_POLICY,
    today: date | None = None,
) -> BuildResult:
    manifests_dir = repo_data_dir / "manifests"
    latest_file = manifests_dir / "LATEST"
    previous_rows: list[ManifestRow] = []
    previous_version = None
    if latest_file.exists():
        previous_version = latest_file.read_text(encoding="utf-8").strip()
        previous_rows = read_manifest(manifests_dir / f"{previous_version}.parquet")
    previous = {r.image_id: r for r in previous_rows}

    rows = load_source_manifests(data_dir)
    if not rows:
        raise RuntimeError("no source manifests found; run `make data-oxford` first")
    rows = add_phashes(rows, data_dir)
    rows = assign_groups(rows, breeds, policy)
    rows = assign_splits(rows, breeds, previous, policy)

    digest = content_hash(rows)
    if previous_version and content_hash(previous_rows) == digest:
        log.info("dataset unchanged: %s", previous_version)
        return BuildResult(previous_version, False, manifests_dir / f"{previous_version}.parquet")

    version = f"{(today or date.today()).isoformat()}-{digest[:8]}"
    rows = [
        replace(r, added_in=previous[r.image_id].added_in if r.image_id in previous else version)
        for r in rows
    ]
    out = manifests_dir / f"{version}.parquet"
    write_manifest(rows, out)
    report = render_report(version, rows, breeds)
    (manifests_dir / f"{version}.md").write_text(report, encoding="utf-8")
    latest_file.write_text(version + "\n", encoding="utf-8")

    local = data_dir / "manifests" / version
    local.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(rows), local / "manifest.parquet")
    (local / "report.md").write_text(report, encoding="utf-8")
    images = data_dir / "processed" / "images"
    with (local / "checksums.sha256").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            p = stored_path(images, r.image_id)
            file_hash = hashlib.sha256(p.read_bytes()).hexdigest()
            fh.write(f"{file_hash}  {p.relative_to(data_dir).as_posix()}\n")
    log.info("dataset %s: %d images", version, len(rows))
    return BuildResult(version, True, out)
