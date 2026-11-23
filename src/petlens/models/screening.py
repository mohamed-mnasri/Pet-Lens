"""Commons screening (DATASET.md, "Human-in-the-loop").

The v0 head (Oxford, my own and licensed photos, and domestic cats) predicts every
Commons and Openverse breed photo in the
train/val/calibration pool. If its top-1 matches the Commons category, the photo is
admitted to training. If not, it waits for a human review. A seeded 5% of the admitted
photos also go to review, to measure how noisy agreed labels are. Top-1, not top-3,
because top-3 would let confusable pairs through (a Birman filed as Ragdoll).

The result is a small committed file per dataset version; nothing here changes the
dataset manifest.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from petlens.data.catalog import BreedCatalog
from petlens.data.manifest import ManifestRow
from petlens.models.heads import train_species_head
from petlens.models.train import POOL, build_arrays

SAMPLE_RATE = 0.05
# Sources whose labels are only weak hints (a Commons category, an Openverse search).
SCREENED_SOURCES = {"commons", "openverse"}
SCHEMA = pa.schema(
    [
        ("image_id", pa.string()),
        ("split", pa.string()),
        ("label", pa.string()),
        ("predicted", pa.string()),
        ("p_label", pa.float32()),
        ("p_predicted", pa.float32()),
        ("decision", pa.string()),  # admit | admit_and_review | review
    ]
)


@dataclass(frozen=True)
class ScreenResult:
    image_id: str
    split: str
    label: str
    predicted: str
    p_label: float
    p_predicted: float
    decision: str


def _sampled(image_id: str, seed: int, rate: float) -> bool:
    h = hashlib.sha256(f"{seed}:review-sample:{image_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < rate


def candidates(rows: list[ManifestRow], breeds: BreedCatalog) -> list[ManifestRow]:
    role = {b.id: b.role for b in breeds.breeds}
    return [
        r
        for r in rows
        if r.source in SCREENED_SOURCES
        and r.breed != "domestic_cat"
        and role.get(r.breed) == "train"
        and (r.split or "") in POOL
    ]


def screen(
    rows: list[ManifestRow],
    emb: dict[str, np.ndarray],
    breeds: BreedCatalog,
    seed: int = 0,
    sample_rate: float = SAMPLE_RATE,
) -> list[ScreenResult]:
    results: list[ScreenResult] = []
    todo = [r for r in candidates(rows, breeds) if r.image_id in emb]
    for species in ("cat", "dog"):
        mine = [r for r in todo if r.species == species]
        if not mine:
            continue
        d = build_arrays(rows, emb, breeds, species)  # v0 pool: nothing admitted yet
        empty_x, empty_y = np.zeros((0, d.x["train"].shape[1])), np.zeros(0, dtype=np.int64)
        head = train_species_head(
            species,
            d.classes,
            d.x["train"],
            d.y["train"],
            d.x.get("val", empty_x),
            d.y.get("val", empty_y),
            d.x.get("calibration", empty_x),
            d.y.get("calibration", empty_y),
            seed=seed,
        )
        # A breed with no trusted photos yet can't be screened: it waits, unused.
        seeded = set(d.y["train"].tolist())
        mine = [r for r in mine if d.classes.index(r.breed) in seeded]
        if not mine:
            continue
        probs = head.predict_proba(np.stack([emb[r.image_id] for r in mine]).astype(np.float64))
        for r, p in zip(mine, probs, strict=True):
            top = int(p.argmax())
            label_idx = d.classes.index(r.breed)
            if top == label_idx:
                decision = (
                    "admit_and_review" if _sampled(r.image_id, seed, sample_rate) else "admit"
                )
            else:
                decision = "review"
            results.append(
                ScreenResult(
                    image_id=r.image_id,
                    split=r.split or "",
                    label=r.breed,
                    predicted=d.classes[top],
                    p_label=float(p[label_idx]),
                    p_predicted=float(p[top]),
                    decision=decision,
                )
            )
    results.sort(key=lambda s: s.image_id)
    return results


def admitted(results: list[ScreenResult]) -> frozenset[str]:
    return frozenset(r.image_id for r in results if r.decision.startswith("admit"))


def write(results: list[ScreenResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([asdict(r) for r in results], schema=SCHEMA), path)


def read(path: Path) -> list[ScreenResult]:
    return [ScreenResult(**rec) for rec in pq.read_table(path, schema=SCHEMA).to_pylist()]


def render_report(version: str, results: list[ScreenResult]) -> str:
    per_label: dict[str, Counter[str]] = {}
    confusions: Counter[tuple[str, str]] = Counter()
    for r in results:
        per_label.setdefault(r.label, Counter())[r.decision] += 1
        if r.decision == "review":
            confusions[(r.label, r.predicted)] += 1
    total = Counter(r.decision for r in results)
    n = len(results) or 1
    admit_n = total["admit"] + total["admit_and_review"]
    lines = [
        f"# Commons screening for dataset {version}",
        "",
        f"- Candidates: {len(results)}",
        f"- Admitted (top-1 agrees with the category): {admit_n} ({admit_n / n:.0%}), "
        f"of which {total['admit_and_review']} also sampled for review",
        f"- Waiting for review (top-1 disagrees): {total['review']} ({total['review'] / n:.0%})",
        "",
        "| Label | Admitted | Sampled for review | Disagreed |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in sorted(per_label):
        c = per_label[label]
        lines.append(
            f"| {label} | {c['admit'] + c['admit_and_review']} | {c['admit_and_review']} | "
            f"{c['review']} |"
        )
    if confusions:
        lines += ["", "Most common disagreements (Commons category -> model's top-1):", ""]
        lines += [f"- {a} -> {b}: {k}" for (a, b), k in confusions.most_common(10)]
    return "\n".join(lines) + "\n"
