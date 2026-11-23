"""The golden set: a small, frozen set of hand-checked test photos for the promotion gate.

Picked once (``make golden``) from held-out photos whose label a person checked: the
Oxford test set (labelled by hand by its authors) and held-out photos labelled by me
(my own photos, my breed folders, and photos I confirmed or corrected in review). Up to
``per_breed`` photos per breed the model knows, chosen by a seeded hash so the pick is
reproducible. Weak photos (black and white, blurry, cut off) are left out.

The file (``data/golden/golden.parquet``) is committed and never rewritten: the labels
in it are the truth for the gate even if a later review changes the manifest. Golden
photos are never trained on: training drops any that end up in the pool.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from petlens.data.catalog import REPO_DATA_DIR, BreedCatalog
from petlens.data.manifest import ManifestRow
from petlens.data.quality import is_weak

GOLDEN_PATH = REPO_DATA_DIR / "golden" / "golden.parquet"
HAND_CHECKED = frozenset({"oxford", "me"})  # label_origin values a person checked
HELD_OUT = ("test_in_domain", "test_unseen_source", "test_domestic")
NEVER = frozenset({"review_unsure", "audit_only", "no_display"})


@dataclass(frozen=True)
class GoldenRow:
    image_id: str
    species: str
    breed: str
    source: str
    label_origin: str
    split: str
    dataset_version: str
    frozen_on: str


def _rank(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"golden:{seed}:{image_id}".encode()).hexdigest()


def select(
    rows: list[ManifestRow],
    breeds: BreedCatalog,
    dataset_version: str,
    per_breed: int = 10,
    seed: int = 0,
) -> list[GoldenRow]:
    known = {s: set(breeds.training_classes(s)) for s in ("cat", "dog")}
    pools: dict[tuple[str, str], list[ManifestRow]] = {}
    for r in rows:
        if (
            r.split in HELD_OUT
            and r.label_origin in HAND_CHECKED
            and r.breed in known.get(r.species, set())
            and not is_weak(r)
            and not NEVER & set(r.flags)
        ):
            pools.setdefault((r.species, r.breed), []).append(r)
    today = date.today().isoformat()
    out = []
    for key in sorted(pools):
        for r in sorted(pools[key], key=lambda r: _rank(seed, r.image_id))[:per_breed]:
            out.append(
                GoldenRow(
                    r.image_id,
                    r.species,
                    r.breed,
                    r.source,
                    r.label_origin,
                    r.split or "",
                    dataset_version,
                    today,
                )
            )
    return out


def write(golden: list[GoldenRow], path: Path = GOLDEN_PATH) -> None:
    if path.exists():
        raise FileExistsError(f"{path} is frozen; delete it by hand to pick a new golden set")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(GoldenRow.__dataclass_fields__)
    table = pa.table({c: pa.array([asdict(g)[c] for g in golden], pa.string()) for c in cols})
    pq.write_table(table, path)


def read(path: Path = GOLDEN_PATH) -> list[GoldenRow]:
    if not path.exists():
        return []
    return [GoldenRow(**d) for d in pq.read_table(path).to_pylist()]


def ids(path: Path = GOLDEN_PATH) -> frozenset[str]:
    return frozenset(g.image_id for g in read(path))


def summary(golden: list[GoldenRow]) -> str:
    if not golden:
        return "no golden set yet; run `make golden`"
    per_species = Counter(g.species for g in golden)
    per_breed = Counter((g.species, g.breed) for g in golden)
    origins = Counter(g.label_origin for g in golden)
    lines = [
        f"golden set: {len(golden)} photos, frozen on {golden[0].frozen_on} "
        f"from {golden[0].dataset_version}",
        "  " + ", ".join(f"{s}: {n}" for s, n in sorted(per_species.items())),
        "  label checked by: " + ", ".join(f"{o} {n}" for o, n in sorted(origins.items())),
    ]
    thin = sorted(f"{b} ({n})" for (_, b), n in per_breed.items() if n < 5)
    if thin:
        lines.append("  fewer than 5 photos: " + ", ".join(thin))
    return "\n".join(lines)
