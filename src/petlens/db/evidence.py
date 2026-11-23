"""The evidence panel: the training photos most similar to a detected animal.

``make index`` copies the training embeddings into ``train_image`` (pgvector); the API
then asks Postgres for the nearest training photos of the suggested breed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import delete, insert, text
from sqlalchemy.orm import Session

from petlens import contracts as c
from petlens.config import Settings
from petlens.db import models as m

EVIDENCE_K = 5


@dataclass(frozen=True)
class TrainRecord:
    image_id: str
    species: str
    breed: str
    source: str
    source_ref: str | None
    licence: str | None
    attribution: str | None
    embedding: list[float]


def replace_index(
    session: Session, records: Iterable[TrainRecord], dataset_version: str, embed_model: str
) -> int:
    """Replace the whole index with ``records`` (one dataset version at a time)."""
    session.execute(delete(m.TrainImage))
    rows = [
        {
            "image_id": r.image_id,
            "dataset_version": dataset_version,
            "species": r.species,
            "breed": r.breed,
            "source": r.source,
            "source_ref": r.source_ref,
            "licence": r.licence,
            "attribution": r.attribution,
            "embedding": r.embedding,
            "embed_model": embed_model,
        }
        for r in records
    ]
    for start in range(0, len(rows), 1000):
        session.execute(insert(m.TrainImage), rows[start : start + 1000])
    return len(rows)


def index_from_files(session: Session, settings: Settings) -> int:
    """Index the training pool of the latest run (the same images the kNN score uses)."""
    from petlens.data.catalog import REPO_DATA_DIR
    from petlens.data.manifest import read_manifest
    from petlens.models import embed, screening
    from petlens.models.train import in_training_pool
    from petlens.vision.load import latest_run

    run = latest_run(settings.models_dir)
    info = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    version, model_name = info["dataset_version"], info["embedding_model"]
    rows = read_manifest(REPO_DATA_DIR / "manifests" / f"{version}.parquet")
    screen_path = REPO_DATA_DIR / "screening" / f"{version}.parquet"
    admitted = (
        screening.admitted(screening.read(screen_path)) if screen_path.exists() else frozenset()
    )
    vecs = embed.read_embeddings(
        embed.cache_path(settings.data_dir, model_name).with_name(f"{version}.parquet")
    )
    records = (
        TrainRecord(
            r.image_id,
            r.species,
            r.breed,
            r.source,
            r.source_ref,
            r.licence,
            r.attribution,
            [float(v) for v in vecs[r.image_id]],
        )
        for r in rows
        if r.split == "train" and r.image_id in vecs and in_training_pool(r, admitted)
    )
    return replace_index(session, records, version, model_name)


_FOR_DETECTION = text(
    "SELECT embedding::text FROM detection WHERE id = :det AND embedding IS NOT NULL"
)
_FOR_GROUP = text(
    """
    SELECT avg(d.embedding)::text FROM detection d JOIN frame f ON f.id = d.frame_id
    WHERE f.upload_id = :upload AND d.group_idx = :group AND d.embedding IS NOT NULL
    """
)
_NEAREST = text(
    """
    SELECT image_id, breed, source, source_ref, licence, attribution,
           1 - (embedding <=> CAST(:q AS vector)) AS similarity
    FROM train_image
    WHERE species = :species AND breed = :breed
    ORDER BY embedding <=> CAST(:q AS vector)
    LIMIT :k
    """
)


def evidence_for(session: Session, p: m.Prediction, k: int = EVIDENCE_K) -> list[c.Evidence]:
    """Nearest training photos of the predicted breed (or of domestic cats)."""
    if p.label is None:
        return []
    q: str | None  # pgvector's text form, "[0.1,0.2,...]"
    if p.detection_id is not None:
        q = session.execute(_FOR_DETECTION, {"det": p.detection_id}).scalar_one_or_none()
    elif p.group_idx is not None:
        q = session.execute(
            _FOR_GROUP, {"upload": p.upload_id, "group": p.group_idx}
        ).scalar_one_or_none()
    else:
        q = None
    if q is None:
        return []
    rows = session.execute(
        _NEAREST, {"q": q, "species": p.species, "breed": p.label, "k": k}
    ).mappings()
    return [
        c.Evidence(
            image_id=r["image_id"],
            breed=r["breed"],
            similarity=round(float(r["similarity"]), 4),
            url=f"/images/train/{r['image_id']}",
            source=r["source"],
            source_ref=r["source_ref"],
            licence=r["licence"],
            attribution=r["attribution"],
        )
        for r in rows
    ]
