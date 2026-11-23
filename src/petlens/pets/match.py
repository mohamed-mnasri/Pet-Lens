"""Match each new animal to a known pet with pgvector nearest-neighbour search.

After an upload is stored, every detection without a pet is matched: animals in the same
appearance group of a clip are one unit (their mean embedding). The unit's nearest
assigned detections of the same species (cosine distance, ``k`` of them) vote; the pet
with the most neighbours within ``max_distance`` wins (ties: the smaller mean distance).
With no neighbour close enough, the animal becomes a new candidate pet. Every assignment
is an ``identity_event`` with origin ``model``.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from petlens.db import models as m


@dataclass(frozen=True)
class MatchConfig:
    max_distance: float = 0.30  # tuned with `make pets-eval` on my labelled photos
    k: int = 5


DEFAULT_MATCH = MatchConfig()


def vote(
    neighbours: list[tuple[uuid.UUID, float]], cfg: MatchConfig
) -> tuple[uuid.UUID | None, float]:
    """The pet most neighbours within ``max_distance`` belong to, and its mean distance."""
    close: dict[uuid.UUID, list[float]] = defaultdict(list)
    for pet, d in neighbours:
        if d <= cfg.max_distance:
            close[pet].append(d)
    if not close:
        return None, min((d for _, d in neighbours), default=float("nan"))
    pet, ds = max(close.items(), key=lambda kv: (len(kv[1]), -float(np.mean(kv[1]))))
    return pet, float(np.mean(ds))


def _unit_vector(vectors: list[list[float]]) -> list[float]:
    v = np.mean(np.asarray(vectors, dtype=np.float64), axis=0)
    n = np.linalg.norm(v)
    return [float(x) for x in (v / n if n else v)]


def units(session: Session, upload_id: uuid.UUID) -> list[list[m.Detection]]:
    q = (
        select(m.Detection)
        .join(m.Frame)
        .where(
            m.Frame.upload_id == upload_id,
            m.Detection.pet_id.is_(None),
            m.Detection.embedding.is_not(None),
        )
        .order_by(m.Frame.frame_idx, m.Detection.created_at)
    )
    groups: dict[object, list[m.Detection]] = {}
    for det in session.scalars(q):
        key = ("group", det.group_idx) if det.group_idx is not None else ("one", det.id)
        groups.setdefault(key, []).append(det)
    return list(groups.values())


def neighbours(
    session: Session, species: str, vector: list[float], k: int
) -> list[tuple[uuid.UUID, float]]:
    dist = m.Detection.embedding.cosine_distance(vector)
    q = (
        select(m.Detection.pet_id, dist)
        .join(m.Pet, m.Pet.id == m.Detection.pet_id)
        .where(m.Detection.species == species, m.Pet.status != "merged")
        .order_by(dist)
        .limit(k)
    )
    return [(pet, float(d)) for pet, d in session.execute(q) if pet is not None]


def assign_upload(
    session: Session, upload_id: uuid.UUID, cfg: MatchConfig = DEFAULT_MATCH
) -> list[uuid.UUID]:
    """Give every animal in the upload a pet. Returns the pet ids, one per unit."""
    out = []
    for unit in units(session, upload_id):
        species = max({d.species for d in unit}, key=[d.species for d in unit].count)
        vector = _unit_vector([list(d.embedding) for d in unit if d.embedding is not None])
        pet_id, distance = vote(neighbours(session, species, vector, cfg.k), cfg)
        action = "assign"
        if pet_id is None:
            pet = m.Pet(species=species)
            session.add(pet)
            session.flush()
            pet_id, action = pet.id, "new_pet"
        for det in unit:
            det.pet_id = pet_id
            session.add(
                m.IdentityEvent(
                    pet_id=pet_id,
                    detection_id=det.id,
                    action=action,
                    origin="model",
                    distance=None if np.isnan(distance) else distance,
                )
            )
        session.flush()
        out.append(pet_id)
    return out
