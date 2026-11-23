"""Serve the next card, record answers as append-only label events, and measure me."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from petlens import contracts as c
from petlens.data.catalog import BreedCatalog, load_breeds
from petlens.db import models as m
from petlens.review.cards import DOMESTIC, NOT_PET

REFERENCE_K = 4


class ReviewError(ValueError):
    pass


@lru_cache(maxsize=1)
def _catalog() -> BreedCatalog:
    return load_breeds()


def nice(label: str) -> str:
    return label.replace("_", " ").title()


def question(card: m.ReviewCard) -> str:
    if card.suggested is None:
        return "What is this? Pick an answer, or say you're not sure."
    if card.suggested == DOMESTIC:
        return "Is this a domestic cat with no clear breed?"
    return f"Is this a {nice(card.suggested)}?"


def references(session: Session, species: str, breed: str | None) -> list[c.Evidence]:
    """A few training photos of the suggested breed, picked the same way each time."""
    if breed is None:
        return []
    rows = session.execute(
        text(
            """
            SELECT image_id, breed, source, source_ref, licence, attribution
            FROM train_image WHERE species = :s AND breed = :b
            ORDER BY md5(image_id) LIMIT :k
            """
        ),
        {"s": species, "b": breed, "k": REFERENCE_K},
    ).mappings()
    return [
        c.Evidence(
            image_id=r["image_id"],
            breed=r["breed"],
            similarity=0.0,
            url=f"/images/train/{r['image_id']}",
            source=r["source"],
            source_ref=r["source_ref"],
            licence=r["licence"],
            attribution=r["attribution"],
        )
        for r in rows
    ]


def picker(card: m.ReviewCard) -> list[str]:
    """Answers offered after "wrong" (or on a blind card): the other top answers, then
    domestic, then "not a cat or dog"."""
    out = [o for o in card.options if o != card.suggested]
    if card.species == "cat" and DOMESTIC not in out and card.suggested != DOMESTIC:
        out.append(DOMESTIC)
    out.append(NOT_PET)
    return out


def _pending(session: Session, queue: str | None, species: str | None = None) -> int:
    q = select(func.count()).select_from(m.ReviewCard).where(m.ReviewCard.state == "pending")
    if queue:
        q = q.where(m.ReviewCard.queue == queue)
    if species:
        q = q.where(m.ReviewCard.species == species)
    return int(session.execute(q).scalar_one())


def next_card(
    session: Session, queue: str | None = None, species: str | None = None
) -> c.CardView | None:
    q = select(m.ReviewCard).where(m.ReviewCard.state == "pending")
    if queue:
        q = q.where(m.ReviewCard.queue == queue)
    if species:
        q = q.where(m.ReviewCard.species == species)
    # Easy questions first; within a priority a stable shuffle, so check cards mix in.
    card = session.scalars(
        q.order_by(m.ReviewCard.priority, func.md5(m.ReviewCard.ref)).limit(1)
    ).first()
    if card is None:
        return None
    box = None
    x0, y0, x1, y1 = card.x0, card.y0, card.x1, card.y1
    if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
        box = (x0, y0, x1, y1)
    return c.CardView(
        id=card.id,
        queue=card.queue,
        image_url=card.image_url,
        box=box,
        species=card.species,
        suggested=card.suggested,
        question=question(card),
        options=picker(card),
        references=references(session, card.species, card.suggested),
        pending=_pending(session, queue, species),
    )


def _allowed(card: m.ReviewCard) -> set[str]:
    breeds = {b.id for b in _catalog().breeds if b.species == card.species}
    return breeds | set(card.options) | {NOT_PET} | ({DOMESTIC} if card.species == "cat" else set())


def answer(session: Session, card_id: uuid.UUID, a: c.ReviewIn) -> m.LabelEvent:
    card = session.get(m.ReviewCard, card_id, with_for_update=True)
    if card is None:
        raise KeyError(card_id)
    if card.state != "pending":
        raise ReviewError("This card was already answered")
    blind = card.suggested is None
    fallback = card.suggested or card.model_label or "unknown"
    action: c.LabelAction
    if a.answer in ("unsure", "skip"):
        action, label = a.answer, fallback
    elif blind or a.answer == "wrong":
        if not a.label:
            raise ReviewError("Pick an answer")
        if a.label not in _allowed(card):
            raise ReviewError(f"Unknown answer: {a.label}")
        label = a.label
        if label == NOT_PET:
            action = "reject"
        elif blind:
            action = "accept" if label == card.model_label else "fix"
        else:
            if label == card.suggested:
                raise ReviewError("That's the suggested answer; swipe right instead")
            action = "fix"
    else:  # correct
        if blind:
            raise ReviewError("Pick an answer")
        action, label = "accept", card.suggested or fallback
    event = m.LabelEvent(
        image_id=card.image_id,
        detection_id=card.detection_id,
        species=card.species,
        proposed_label=label,
        origin="me",
        action=action,
        reviewer="me",
        review_ms=a.review_ms,
        review_card_id=card.id,
    )
    session.add(event)
    card.state = "done"
    card.answered_at = datetime.now(UTC)
    session.flush()
    return event


def check_is_right(card: m.ReviewCard, e: m.LabelEvent) -> bool | None:
    """Was my answer on a check card right? None when I said "not sure" or skipped."""
    if e.action in ("unsure", "skip"):
        return None
    if e.action == "accept":
        return card.suggested == card.truth
    return e.proposed_label == card.truth


def stats(session: Session) -> c.ReviewStats:
    pending = dict(
        session.execute(
            select(m.ReviewCard.queue, func.count())
            .where(m.ReviewCard.state == "pending")
            .group_by(m.ReviewCard.queue)
        ).all()
    )
    answered = int(
        session.execute(
            select(func.count()).select_from(m.LabelEvent).where(m.LabelEvent.origin == "me")
        ).scalar_one()
    )
    checks = session.execute(
        select(m.ReviewCard, m.LabelEvent)
        .join(m.LabelEvent, m.LabelEvent.review_card_id == m.ReviewCard.id)
        .where(m.ReviewCard.queue == "check")
    ).all()
    verdicts = [v for card, e in checks if (v := check_is_right(card, e)) is not None]
    right = sum(verdicts)
    return c.ReviewStats(
        pending={str(k): int(v) for k, v in pending.items()},
        answered=answered,
        checks_answered=len(verdicts),
        checks_correct=right,
        reviewer_accuracy=round(right / len(verdicts), 3) if verdicts else None,
    )
