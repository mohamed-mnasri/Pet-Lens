"""Step 15: matching animals to pets, my changes, profiles and the re-ID report."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import numpy as np
import pytest

from petlens.pets import evaluate
from petlens.pets.match import MatchConfig, vote

DATABASE_URL = os.environ.get("DATABASE_URL")
A, B = uuid.UUID(int=1), uuid.UUID(int=2)


def test_vote_needs_close_neighbours_and_prefers_the_majority() -> None:
    cfg = MatchConfig(max_distance=0.3, k=5)
    assert vote([(A, 0.1), (B, 0.05), (A, 0.2)], cfg) == (A, pytest.approx(0.15))
    assert vote([(A, 0.5), (B, 0.6)], cfg)[0] is None  # nobody close: a new pet
    assert vote([], cfg)[0] is None


def _items(seed: int = 0) -> list[evaluate.Labelled]:
    rng = np.random.default_rng(seed)
    out = []
    for truth, centre in ((A, rng.normal(size=32)), (B, rng.normal(size=32))):
        for _ in range(4):
            v = centre + rng.normal(size=32) * 0.15
            out.append(
                evaluate.Labelled(
                    uuid.uuid4(), (uuid.uuid4(),), "cat", v / np.linalg.norm(v), truth
                )
            )
    return out


def test_replay_scores_pairs() -> None:
    items = _items()
    p, r, pairs = evaluate.pairwise(items, evaluate.replay(items, MatchConfig(max_distance=0.3)))
    assert (p, r, pairs) == (1.0, 1.0, 12)
    lumped = evaluate.replay(items, MatchConfig(max_distance=1.9))  # everything is one pet
    p, r, _ = evaluate.pairwise(items, lumped)
    assert r == 1.0 and p < 0.5
    names = {A: "Milo", B: "Luna"}
    assert any("mixed" in line for line in evaluate.failures(items, lumped, names))


@pytest.fixture
def session() -> Iterator[object]:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from petlens.db import migrate

    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    migrate.upgrade(DATABASE_URL)
    with Session(engine) as s:
        yield s
    engine.dispose()


def _upload(s: object, vectors: list[np.ndarray], group: int | None = None) -> uuid.UUID:
    from petlens.db import models as m

    up = m.Upload(
        kind="photo", original_name="x.jpg", mime="image/jpeg", size_bytes=1, sha256="0" * 64
    )
    s.add(up)  # type: ignore[attr-defined]
    s.flush()  # type: ignore[attr-defined]
    for i, v in enumerate(vectors):
        frame = m.Frame(
            upload_id=up.id,
            frame_idx=i,
            image_id=f"{i}" * 64,
            path="p",
            width=1,
            height=1,
            size_bytes=1,
        )
        s.add(frame)  # type: ignore[attr-defined]
        s.flush()  # type: ignore[attr-defined]
        unit = v / np.linalg.norm(v)
        s.add(  # type: ignore[attr-defined]
            m.Detection(
                frame_id=frame.id,
                species="cat",
                score=0.9,
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                embedding=[float(x) for x in unit],
                group_idx=group,
                crop_path=f"/x/{'a' * 64}.webp",
            )
        )
    s.flush()  # type: ignore[attr-defined]
    return up.id


@pytest.mark.db
@pytest.mark.skipif(not DATABASE_URL, reason="DATABASE_URL not set")
def test_matching_names_merges_splits_and_profiles(session: object) -> None:
    from sqlalchemy import func, select, text
    from sqlalchemy.exc import DBAPIError

    from petlens.db import models as m
    from petlens.pets import manage, match, profiles

    s = session
    rng = np.random.default_rng(1)
    milo, luna = rng.normal(size=384), rng.normal(size=384)
    near = lambda c: c + rng.normal(size=384) * 0.1  # noqa: E731

    first = match.assign_upload(s, _upload(s, [near(milo)]))  # type: ignore[arg-type]
    again = match.assign_upload(s, _upload(s, [near(milo)]))  # type: ignore[arg-type]
    other = match.assign_upload(s, _upload(s, [near(luna)]))  # type: ignore[arg-type]
    clip = match.assign_upload(s, _upload(s, [near(luna), near(luna)], group=0))  # type: ignore[arg-type]
    assert first == again and other == clip and first != other
    assert len(clip) == 1  # one clip group is one unit

    manage.name(s, first[0], "  Milo ")  # type: ignore[arg-type]
    pets = profiles.list_pets(s)  # type: ignore[arg-type]
    assert pets[0]["name"] == "Milo" and pets[0]["photos"] == 2
    assert pets[1]["photos"] == 3 and pets[1]["name"] is None

    # Luna's photos split off, then merged back.
    dets = list(s.scalars(select(m.Detection.id).where(m.Detection.pet_id == other[0])))  # type: ignore[attr-defined]
    new = manage.split(s, other[0], dets[:1])  # type: ignore[arg-type]
    assert profiles.pet_detail(s, new.id)["photos"] == 1  # type: ignore[arg-type,index]
    manage.merge(s, new.id, other[0])  # type: ignore[arg-type]
    assert profiles.pet_detail(s, other[0])["photos"] == 3  # type: ignore[arg-type,index]
    with pytest.raises(manage.PetError):
        manage.name(s, new.id, "x")  # type: ignore[arg-type]  # merged pets are read-only
    manage.confirm(s, dets[0])  # type: ignore[arg-type]
    actions = [e for (e,) in s.execute(select(m.IdentityEvent.action).order_by(m.IdentityEvent.id))]  # type: ignore[attr-defined]
    assert actions[:2] == ["new_pet", "assign"] and {"name", "split", "merge", "confirm"} <= set(
        actions
    )
    s.commit()  # type: ignore[attr-defined]
    with pytest.raises(DBAPIError):  # append-only
        s.execute(text("DELETE FROM identity_event"))  # type: ignore[attr-defined]
    s.rollback()  # type: ignore[attr-defined]
    assert s.scalar(select(func.count()).select_from(m.Pet)) == 3  # type: ignore[attr-defined]

    report = evaluate.report(s)  # type: ignore[arg-type]
    assert "Labelled photos: 3 of 2 animals" in report and "(in use)" in report
