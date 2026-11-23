"""My changes to pets: name, confirm, move, merge and split. Each is an identity_event
with origin ``me``; ``detection.pet_id`` and ``pet`` hold the current state."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from petlens.db import models as m


class PetError(ValueError):
    pass


def _pet(session: Session, pet_id: uuid.UUID) -> m.Pet:
    pet = session.get(m.Pet, pet_id)
    if pet is None:
        raise KeyError(pet_id)
    if pet.status == "merged":
        raise PetError("that pet was merged into another one")
    return pet


def _detection(session: Session, detection_id: uuid.UUID) -> m.Detection:
    det = session.get(m.Detection, detection_id)
    if det is None:
        raise KeyError(detection_id)
    return det


def name(session: Session, pet_id: uuid.UUID, new_name: str) -> m.Pet:
    pet = _pet(session, pet_id)
    clean = " ".join(new_name.split())[:64]
    if not clean:
        raise PetError("the name is empty")
    pet.name, pet.status = clean, "named"
    session.add(m.IdentityEvent(pet_id=pet.id, action="name", origin="me", value=clean))
    session.flush()
    return pet


def confirm(session: Session, detection_id: uuid.UUID) -> None:
    det = _detection(session, detection_id)
    if det.pet_id is None:
        raise PetError("this animal has no pet yet")
    session.add(
        m.IdentityEvent(pet_id=det.pet_id, detection_id=det.id, action="confirm", origin="me")
    )
    session.flush()


def move(session: Session, detection_id: uuid.UUID, pet_id: uuid.UUID) -> None:
    det = _detection(session, detection_id)
    pet = _pet(session, pet_id)
    if pet.species != det.species:
        raise PetError(f"that pet is a {pet.species}, this animal a {det.species}")
    det.pet_id = pet.id
    session.add(m.IdentityEvent(pet_id=pet.id, detection_id=det.id, action="move", origin="me"))
    session.flush()


def merge(session: Session, source: uuid.UUID, target: uuid.UUID) -> m.Pet:
    """All of ``source``'s photos move to ``target``; ``source`` is kept as merged."""
    if source == target:
        raise PetError("pick two different pets")
    src, dst = _pet(session, source), _pet(session, target)
    if src.species != dst.species:
        raise PetError("a cat and a dog can't be the same pet")
    for det in session.scalars(select(m.Detection).where(m.Detection.pet_id == src.id)):
        det.pet_id = dst.id
    src.status, src.merged_into = "merged", dst.id
    if dst.name is None and src.name:
        dst.name, dst.status = src.name, "named"
    session.add(m.IdentityEvent(pet_id=dst.id, other_pet_id=src.id, action="merge", origin="me"))
    session.flush()
    return dst


def split(session: Session, pet_id: uuid.UUID, detection_ids: list[uuid.UUID]) -> m.Pet:
    """Move some of a pet's photos to a new pet."""
    old = _pet(session, pet_id)
    dets = [_detection(session, d) for d in detection_ids]
    if not dets or any(d.pet_id != old.id for d in dets):
        raise PetError("pick photos of this pet to split off")
    new = m.Pet(species=old.species)
    session.add(new)
    session.flush()
    session.add(m.IdentityEvent(pet_id=new.id, other_pet_id=old.id, action="split", origin="me"))
    for det in dets:
        det.pet_id = new.id
        session.add(m.IdentityEvent(pet_id=new.id, detection_id=det.id, action="move", origin="me"))
    session.flush()
    return new
