"""Small fake data for ``dbt build`` in CI: ``python -m petlens.eval.fixtures``.

Resets the database (drops the public and dbt schemas), migrates, and writes an
evaluation of a tiny synthetic run plus a few uploads, predictions, review cards and
answers, so every mart has rows. Never point it at a database you care about.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from petlens.config import get_settings
from petlens.data.catalog import load_breeds
from petlens.data.manifest import ManifestRow
from petlens.db import migrate
from petlens.db import models as m
from petlens.eval import compute, store
from petlens.models import golden, policy, train

DBT_SCHEMAS = ("analytics", "staging", "intermediate", "marts", "snapshots")


def synthetic_rows(
    seed: int = 1, noise: float = 1.0
) -> tuple[list[ManifestRow], dict[str, np.ndarray]]:
    breeds = load_breeds()
    rng = np.random.default_rng(seed)
    rows: list[ManifestRow] = []
    emb: dict[str, np.ndarray] = {}
    splits = ["train"] * 12 + ["val"] * 3 + ["calibration"] * 3 + ["test_in_domain"] * 4
    for species in ("cat", "dog"):
        for breed in breeds.training_classes(species):
            centre = rng.normal(size=16) * 4
            for i, split in enumerate(splits):
                image_id = f"{breed}-{i}".ljust(64, "0")[:64]
                source = "commons" if breed == "domestic_cat" else "oxford"
                if breed == "domestic_cat" and split == "test_in_domain":
                    split = "test_domestic"
                rows.append(
                    ManifestRow(
                        image_id=image_id,
                        source=source,
                        source_ref=image_id,
                        licence="x",
                        attribution="x",
                        species=species,
                        breed=breed,
                        label_origin=source,
                        width=1,
                        height=1,
                        split=split,
                        flags=["blurry"] if i % 7 == 0 else [],
                    )
                )
                emb[image_id] = (centre + noise * rng.normal(size=16)).astype(np.float32)
    return rows, emb


def main() -> None:
    url = get_settings().database_url
    engine = create_engine(url)
    with engine.begin() as conn:
        for schema in DBT_SCHEMAS:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    migrate.upgrade(url)
    rows, emb = synthetic_rows()
    breeds = load_breeds()
    heads, metrics = train.train(rows, emb, breeds)
    cfg = policy.PolicyConfig.model_validate(metrics["policy_config"])
    gold = golden.select(rows, breeds, "fixture", per_breed=2)
    result = compute.evaluate(rows, emb, heads, breeds, cfg, golden=gold)
    with Session(engine) as s:
        store.save(s, result, "fixture-run", "fixture", "fake", compute.composition(rows))
        s.add(m.DatasetVersion(version="fixture", manifest_hash="0" * 64))
        s.flush()
        s.add(
            m.ModelRegistry(
                run_id="fixture-run",
                weights_hash="0" * 64,
                dataset_version="fixture",
                config={"embedding_model": "fake"},
                champion=True,
                gate={"passed": True, "numbers": {"golden set": {"diff": 0.02}}},
            )
        )
        for i, outcome in enumerate(["accept", "no_clear_breed", "review"]):
            up = m.Upload(
                kind="photo",
                original_name=f"{i}.jpg",
                mime="image/jpeg",
                size_bytes=1,
                sha256=f"{i}" * 64,
                status="done",
            )
            s.add(up)
            s.flush()
            s.add(
                m.Prediction(
                    upload_id=up.id,
                    group_idx=0,
                    model_version="fixture",
                    species="cat",
                    top3=[],
                    knn_distance=0.2,
                    outcome=outcome,
                    confidence=0.9,
                    label="ragdoll" if outcome == "accept" else None,
                )
            )
        for i, status in enumerate(["ok", "ok", "parse_failed"]):
            s.add(
                m.VlmTag(
                    image_id=f"{i}".ljust(64, "v"),
                    prompt_version="screen-v1",
                    model_tag="fixture-vl",
                    model_digest="sha256:0",
                    status=status,
                    kind="photo" if status == "ok" else None,
                    whole_animal=i == 0 if status == "ok" else None,
                    kitten=False if status == "ok" else None,
                    blurry=False if status == "ok" else None,
                    several_animals=False if status == "ok" else None,
                    attempts=1 + (status != "ok"),
                    seconds=2.5,
                    raw="{}",
                )
            )
        for i, (queue, action, label, truth) in enumerate(
            [
                ("commons", "accept", "bengal", None),
                ("commons", "unsure", "bengal", None),
                ("check", "accept", "birman", "birman"),
                ("check", "fix", "persian", "ragdoll"),
            ]
        ):
            card = m.ReviewCard(
                ref=f"fixture:{i}",
                queue=queue,
                image_url="/images/train/x",
                species="cat",
                suggested=label,
                options=[],
                truth=truth,
                priority=1,
                state="done",
            )
            s.add(card)
            s.flush()
            s.add(
                m.LabelEvent(
                    image_id=f"{i}".ljust(64, "a"),
                    species="cat",
                    proposed_label=label,
                    origin="me",
                    action=action,
                    review_ms=1000 * (i + 1),
                    review_card_id=card.id,
                )
            )
        s.commit()
    engine.dispose()
    print(f"fixtures: {len(result.images)} evaluated images, 3 predictions, 4 answers")


if __name__ == "__main__":
    main()
