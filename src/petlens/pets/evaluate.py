"""Re-identification precision and recall on my labelled photos (``make pets-eval``).

Truth: the photos I vouched for, meaning photos of pets I named, plus photos I
confirmed, moved or split off. Their current pet is the right answer.

The matcher is replayed on those photos in upload order, at several distance
thresholds, with the same voting as live matching. Pairwise scores over every pair of
labelled photos: precision is the share of pairs put in the same pet that really are
the same animal; recall is the share of same-animal pairs that were put together. The
report lists the threshold in use, the best one, and where it goes wrong (animals split
across pets, pets that mix animals).
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from petlens.db import models as m
from petlens.pets.match import DEFAULT_MATCH, MatchConfig, vote

THRESHOLDS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)


@dataclass
class Labelled:
    detection_id: uuid.UUID
    unit: tuple[Any, ...]  # photos of one clip group are matched together
    species: str
    vector: np.ndarray
    truth: uuid.UUID


def labelled(session: Session) -> list[Labelled]:
    vouched = set(
        session.scalars(
            select(m.IdentityEvent.detection_id).where(
                m.IdentityEvent.origin == "me",
                m.IdentityEvent.action.in_(("confirm", "move")),
                m.IdentityEvent.detection_id.is_not(None),
            )
        )
    )
    named = set(session.scalars(select(m.Pet.id).where(m.Pet.status == "named")))
    q = (
        select(m.Detection, m.Frame.upload_id)
        .join(m.Frame)
        .where(m.Detection.pet_id.is_not(None), m.Detection.embedding.is_not(None))
        .order_by(m.Detection.created_at)
    )
    out = []
    for det, upload in session.execute(q):
        if det.id in vouched or det.pet_id in named:
            assert det.pet_id is not None and det.embedding is not None
            unit = (upload, det.group_idx) if det.group_idx is not None else (det.id,)
            vec = np.asarray(det.embedding, dtype=np.float64)
            out.append(Labelled(det.id, unit, det.species, vec / np.linalg.norm(vec), det.pet_id))
    return out


def replay(items: list[Labelled], cfg: MatchConfig) -> dict[uuid.UUID, int]:
    """Assign photos in order, as the live matcher would. Returns photo -> predicted pet."""
    units: dict[tuple[Any, ...], list[Labelled]] = {}
    for it in items:
        units.setdefault(it.unit, []).append(it)
    assigned: list[tuple[str, np.ndarray, int]] = []
    out: dict[uuid.UUID, int] = {}
    next_pet = 0
    for unit in units.values():
        species = unit[0].species
        v = np.mean([u.vector for u in unit], axis=0)
        v /= np.linalg.norm(v)
        same = [(pet, 1 - float(vec @ v)) for s, vec, pet in assigned if s == species]
        near = sorted(same, key=lambda t: t[1])[: cfg.k]
        pet, _ = vote([(uuid.UUID(int=p), d) for p, d in near], cfg)
        if pet is None:
            pid, next_pet = next_pet, next_pet + 1
        else:
            pid = pet.int
        for u in unit:
            assigned.append((species, u.vector, pid))
            out[u.detection_id] = pid
    return out


def pairwise(items: list[Labelled], predicted: dict[uuid.UUID, int]) -> tuple[float, float, int]:
    tp = fp = fn = 0
    for a, b in combinations(items, 2):
        same_truth = a.truth == b.truth
        same_pred = predicted[a.detection_id] == predicted[b.detection_id]
        tp += same_truth and same_pred
        fp += same_pred and not same_truth
        fn += same_truth and not same_pred
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return precision, recall, tp + fn


def failures(
    items: list[Labelled], predicted: dict[uuid.UUID, int], names: dict[uuid.UUID, str]
) -> list[str]:
    by_truth: dict[uuid.UUID, Counter[int]] = defaultdict(Counter)
    by_pred: dict[int, Counter[uuid.UUID]] = defaultdict(Counter)
    for it in items:
        by_truth[it.truth][predicted[it.detection_id]] += 1
        by_pred[predicted[it.detection_id]][it.truth] += 1
    lines = []
    for truth, preds in by_truth.items():
        if len(preds) > 1:
            lines.append(
                f"- {names.get(truth, 'a pet')} was split into {len(preds)} pets "
                f"({sum(preds.values())} photos)"
            )
    for _, truths in by_pred.items():
        if len(truths) > 1:
            who = ", ".join(names.get(t, "a pet") for t in truths)
            lines.append(f"- one pet mixed {who}")
    return lines


def report(session: Session, current: MatchConfig = DEFAULT_MATCH) -> str:
    items = labelled(session)
    names = {p.id: p.name or "an unnamed pet" for p in session.scalars(select(m.Pet))}
    animals = len({it.truth for it in items})
    lines = [
        "# Pet re-identification on my labelled photos",
        "",
        f"- Labelled photos: {len(items)} of {animals} animals (photos of named pets, and "
        "photos I confirmed, moved or split off)",
        f"- Matching: the {current.k} nearest photos vote; a pet needs a neighbour within "
        "the threshold",
        "",
    ]
    if len(items) < 2 or animals < 1:
        return (
            "\n".join([*lines, "Not enough labelled photos yet: name your pets on the Pets tab."])
            + "\n"
        )
    lines += [
        "| Threshold | Precision | Recall | Same-animal pairs |",
        "| ---: | ---: | ---: | ---: |",
    ]
    best: tuple[float, float] | None = None
    for t in sorted({*THRESHOLDS, current.max_distance}):
        cfg = MatchConfig(max_distance=t, k=current.k)
        p, r, pairs = pairwise(items, replay(items, cfg))
        mark = " (in use)" if t == current.max_distance else ""
        lines.append(f"| {t:.2f}{mark} | {p:.1%} | {r:.1%} | {pairs} |")
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if best is None or f1 > best[0]:
            best = (f1, t)
    assert best is not None
    lines += ["", f"Best F1 at threshold {best[1]:.2f}."]
    fails = failures(items, replay(items, current), names)
    lines += [
        "",
        "Where the threshold in use goes wrong:",
        "",
        *(fails or ["- nowhere on these photos"]),
    ]
    return "\n".join(lines) + "\n"
