"""What the Pets page shows: photos, first and last seen, and the consensus breed.

The consensus breed is the most common top-1 answer over a pet's photos (each photo's
own prediction, or its clip group's), and *consistency* is the share of photos that
agree with it: how consistently the model agrees with itself about one animal.
"""

from __future__ import annotations

import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from petlens.db import models as m


def _top1(top3: list[dict[str, Any]] | None) -> str | None:
    return str(top3[0]["label"]) if top3 else None


def photos(session: Session, pet_ids: list[uuid.UUID]) -> dict[uuid.UUID, list[dict[str, Any]]]:
    """Each pet's photos, newest first, with the model's top-1 answer for each."""
    pred = m.Prediction
    q = (
        select(m.Detection, m.Frame.upload_id, m.Upload.created_at, pred.top3)
        .join(m.Frame, m.Frame.id == m.Detection.frame_id)
        .join(m.Upload, m.Upload.id == m.Frame.upload_id)
        .outerjoin(
            pred,
            or_(
                pred.detection_id == m.Detection.id,
                and_(
                    pred.detection_id.is_(None),
                    pred.upload_id == m.Frame.upload_id,
                    pred.group_idx == m.Detection.group_idx,
                ),
            ),
        )
        .where(m.Detection.pet_id.in_(pet_ids))
        .order_by(m.Upload.created_at.desc(), m.Detection.created_at)
    )
    out: dict[uuid.UUID, list[dict[str, Any]]] = {p: [] for p in pet_ids}
    seen: set[uuid.UUID] = set()
    for det, upload_id, at, top3 in session.execute(q):
        if det.id in seen or det.pet_id is None:
            continue
        seen.add(det.id)
        out[det.pet_id].append(
            {
                "detection_id": str(det.id),
                "upload_id": str(upload_id),
                "seen_at": at,
                "crop": Path(det.crop_path).stem if det.crop_path else None,
                "top1": _top1(top3),
            }
        )
    return out


def summary(pet: m.Pet, shots: list[dict[str, Any]]) -> dict[str, Any]:
    answers = Counter(s["top1"] for s in shots if s["top1"])
    breed, agree = answers.most_common(1)[0] if answers else (None, 0)
    return {
        "id": str(pet.id),
        "name": pet.name,
        "species": pet.species,
        "status": pet.status,
        "photos": len(shots),
        "first_seen": min((s["seen_at"] for s in shots), default=None),
        "last_seen": max((s["seen_at"] for s in shots), default=None),
        "consensus_breed": breed,
        "consistency": agree / sum(answers.values()) if answers else None,
        "crops": [s["crop"] for s in shots if s["crop"]][:6],
    }


def list_pets(session: Session) -> list[dict[str, Any]]:
    pets = list(session.scalars(select(m.Pet).where(m.Pet.status != "merged")))
    shots = photos(session, [p.id for p in pets])
    rows = [summary(p, shots[p.id]) for p in pets]
    # Named pets first, then candidates with the most photos.
    return sorted(rows, key=lambda r: (r["name"] is None, -r["photos"], r["id"]))


def pet_detail(session: Session, pet_id: uuid.UUID) -> dict[str, Any] | None:
    pet = session.get(m.Pet, pet_id)
    if pet is None:
        return None
    shots = photos(session, [pet.id])[pet.id]
    return summary(pet, shots) | {"all_photos": shots}
