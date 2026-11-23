"""Build review cards for the four queues.

- ``commons``: Commons photos screening disagreed with, plus the 5% sample of agreements.
- ``batch``: my own photos and clips from review batches, one card per animal.
- ``audit``: a seeded 5% of confident answers on my batches, shown without the answer.
- ``check``: Oxford test photos with known labels, about 5% of cards, to measure me.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from petlens.data.catalog import BreedCatalog
from petlens.data.manifest import ManifestRow
from petlens.db import models as m
from petlens.models.screening import ScreenResult

AUDIT_RATE = 0.05
CHECK_RATE = 0.05
DOMESTIC = "domestic_cat"
NOT_PET = "not_pet"

# Lower runs first. Questions I can answer reliably (is it a domestic cat?) come before
# fine breed calls.
PRIORITY_EASY = 0
PRIORITY_NORMAL = 1
PRIORITY_SAMPLE = 2


def unit(seed: int, salt: str, key: str) -> float:
    h = hashlib.sha256(f"{seed}:{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _box(r: ManifestRow) -> dict[str, float | None]:
    keys = ("x0", "y0", "x1", "y1")
    if r.bbox and len(r.bbox) == 4:
        return dict(zip(keys, (float(v) for v in r.bbox), strict=True))
    return dict.fromkeys(keys)


def commons_cards(
    rows: Iterable[ManifestRow], results: Iterable[ScreenResult]
) -> list[dict[str, Any]]:
    by_id = {r.image_id: r for r in rows}
    cards = []
    for s in results:
        if s.decision not in ("review", "admit_and_review") or s.image_id not in by_id:
            continue
        r = by_id[s.image_id]
        disagreed = s.decision == "review"
        if not disagreed:
            priority = PRIORITY_SAMPLE
        elif s.predicted == DOMESTIC:
            priority = PRIORITY_EASY  # "is this really a <breed>, or a domestic cat?"
        else:
            priority = PRIORITY_NORMAL
        options = [s.predicted] if s.predicted != s.label else []
        cards.append(
            {
                "ref": f"commons:{s.image_id}",
                "queue": "commons",
                "image_id": s.image_id,
                "image_url": f"/images/train/{s.image_id}",
                **_box(r),
                "species": r.species,
                "suggested": s.label,  # the Commons category's breed
                "options": options,
                "model_label": s.predicted,
                "priority": priority,
            }
        )
    return cards


def check_cards(
    rows: Iterable[ManifestRow], breeds: BreedCatalog, n: int, seed: int = 0
) -> list[dict[str, Any]]:
    """``n`` Oxford test photos. Half show the true breed, half a wrong one of the same
    species, so "correct" isn't always the right answer."""
    tests = sorted(
        (r for r in rows if r.source == "oxford" and r.split == "test_in_domain"),
        key=lambda r: unit(seed, "check", r.image_id),
    )[:n]
    cards = []
    for r in tests:
        classes = [c for c in breeds.training_classes(r.species) if c != r.breed]  # type: ignore[arg-type]
        show_truth = unit(seed, "check-truth", r.image_id) < 0.5
        if show_truth or not classes:
            suggested = r.breed
        else:
            suggested = classes[int(unit(seed, "check-wrong", r.image_id) * len(classes))]
        cards.append(
            {
                "ref": f"check:{r.image_id}",
                "queue": "check",
                "image_id": r.image_id,
                "image_url": f"/images/train/{r.image_id}",
                **_box(r),
                "species": r.species,
                "suggested": suggested,
                "options": [r.breed] if suggested != r.breed else [],
                "truth": r.breed,
                "priority": PRIORITY_NORMAL,
            }
        )
    return cards


def insert_cards(session: Session, cards: list[dict[str, Any]]) -> int:
    """Add cards that don't exist yet (by ``ref``). Returns how many were new."""
    if not cards:
        return 0
    added = 0
    for start in range(0, len(cards), 500):
        stmt = (
            insert(m.ReviewCard)
            .values(cards[start : start + 500])
            .on_conflict_do_nothing(index_elements=["ref"])
            .returning(m.ReviewCard.id)
        )
        added += len(session.execute(stmt).all())
    return added


def cards_for_upload(
    session: Session, upload_id: uuid.UUID, audit_rate: float = AUDIT_RATE, seed: int = 0
) -> int:
    """One card per animal in a batch upload: per detection for photos and ambiguous clips,
    per appearance group for clips. Confident answers are sometimes held back as a blind
    audit."""
    preds = session.scalars(select(m.Prediction).where(m.Prediction.upload_id == upload_id)).all()
    dets = session.scalars(
        select(m.Detection)
        .join(m.Frame)
        .where(m.Frame.upload_id == upload_id)
        .order_by(m.Frame.frame_idx)
    ).all()
    frames = {
        f.id: f for f in session.scalars(select(m.Frame).where(m.Frame.upload_id == upload_id))
    }
    cards = []
    for p in preds:
        if p.detection_id is not None:
            det = next((d for d in dets if d.id == p.detection_id), None)
        else:
            det = next((d for d in dets if d.group_idx == p.group_idx), None)
        if det is None:
            continue
        frame = frames[det.frame_id]
        top = [t["label"] for t in p.top3]
        model_label = p.label if p.label else (top[0] if top else None)
        key = f"{upload_id}:{det.id}"
        blind = p.outcome == "accept" and unit(seed, "audit", key) < audit_rate
        options = [t for t in top if t != model_label] if not blind else top
        if p.species == "cat" and DOMESTIC not in options and model_label != DOMESTIC:
            options.append(DOMESTIC)
        cards.append(
            {
                "ref": f"{'audit' if blind else 'batch'}:{det.id}",
                "queue": "audit" if blind else "batch",
                "image_id": frame.image_id,
                "detection_id": det.id,
                "upload_id": upload_id,
                "image_url": f"/images/frames/{frame.image_id}",
                "x0": det.x0,
                "y0": det.y0,
                "x1": det.x1,
                "y1": det.y1,
                "species": p.species,
                "suggested": None if blind else model_label,
                "options": options,
                "model_label": model_label,
                "priority": PRIORITY_EASY if model_label == DOMESTIC else PRIORITY_NORMAL,
            }
        )
    return insert_cards(session, cards)
