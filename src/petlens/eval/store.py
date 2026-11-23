"""Write an evaluation into the metric tables (``eval_*``, ``dataset_count``)."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import delete, insert
from sqlalchemy.orm import Session

from petlens.db import models as m
from petlens.eval.compute import EvalResult


def save(
    session: Session,
    result: EvalResult,
    run_id: str,
    dataset_version: str,
    embedding_model: str,
    composition: list[dict[str, Any]],
) -> uuid.UUID:
    run = m.EvalRun(run_id=run_id, dataset_version=dataset_version, embedding_model=embedding_model)
    session.add(run)
    session.flush()
    for table, rows in (
        (m.EvalMetric, result.metrics),
        (m.EvalClassMetric, result.class_metrics),
        (m.EvalImage, result.images),
    ):
        for start in range(0, len(rows), 2000):
            chunk = [r | {"eval_run_id": run.id} for r in rows[start : start + 2000]]
            session.execute(insert(table), chunk)
    session.execute(delete(m.DatasetCount).where(m.DatasetCount.dataset_version == dataset_version))
    if composition:
        session.execute(
            insert(m.DatasetCount), [c | {"dataset_version": dataset_version} for c in composition]
        )
    return run.id
