"""The model registry: every trained run, its weights hash, data, config and metrics, and
which one is the champion the app serves.

``make register`` records the latest run. ``make gate`` compares it with the champion
(``petlens.models.promotion``) and ``make promote`` does the same and, if the gate
passes, makes it the champion. The first registered run becomes champion without a
comparison. The API and the evidence index serve the champion; with no champion (or no
database) they fall back to the latest run.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from petlens.data.catalog import REPO_DATA_DIR
from petlens.db import models as m
from petlens.models.runs import LoadedRun, all_runs, latest_run

log = logging.getLogger(__name__)


def _config(run: LoadedRun) -> dict[str, Any]:
    metrics = run.info.get("metrics", {})
    return {
        "embedding_model": run.embedding_model,
        "heads": {s: h.kind for s, h in run.heads.items()},
        "temperatures": {s: h.temperature for s, h in run.heads.items()},
        "classes": {s: len(h.classes) for s, h in run.heads.items()},
        "policy": run.cfg.model_dump(),
        "finetune": metrics.get("finetune"),
    }


def _summary(run: LoadedRun) -> dict[str, Any]:
    """The headline numbers from the training report, per species and test set."""
    out: dict[str, Any] = {}
    for species, mt in run.info.get("metrics", {}).items():
        if not isinstance(mt, dict) or "tests" not in mt:
            continue
        entry: dict[str, Any] = {
            test: round(t["calibrated"]["top1"], 4) for test, t in mt["tests"].items()
        }
        pol = mt.get("policy") or {}
        if "test_domestic" in pol:
            entry["false_breed_claim_rate"] = pol["test_domestic"]["false_breed_claim_rate"]
        if "test_unknown" in pol:
            entry["unknown_rejection_rate"] = pol["test_unknown"]["unknown_rejection_rate"]
        out[species] = entry
    return out


def ensure_dataset_version(session: Session, version: str, repo_data: Path = REPO_DATA_DIR) -> None:
    if session.get(m.DatasetVersion, version) is not None:
        return
    path = repo_data / "manifests" / f"{version}.parquet"
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
    counts: dict[str, Any] = {}
    if path.exists():
        from petlens.data.manifest import read_manifest

        counts = dict(Counter(r.split or "" for r in read_manifest(path)))
    session.add(m.DatasetVersion(version=version, manifest_hash=digest, counts=counts))
    session.flush()


def register(session: Session, run: LoadedRun) -> m.ModelRegistry:
    """Record a run (idempotent: the same run id is updated, never duplicated)."""
    ensure_dataset_version(session, run.dataset_version)
    entry = session.scalars(
        select(m.ModelRegistry).where(m.ModelRegistry.run_id == run.run_id)
    ).one_or_none()
    if entry is None:
        entry = m.ModelRegistry(run_id=run.run_id)
        session.add(entry)
    entry.weights_hash = str(run.info.get("weights_sha256", ""))
    entry.dataset_version = run.dataset_version
    entry.config = _config(run)
    entry.metrics = _summary(run)
    session.flush()
    return entry


def champion(session: Session) -> m.ModelRegistry | None:
    return session.scalars(select(m.ModelRegistry).where(m.ModelRegistry.champion)).one_or_none()


def record_gate(
    session: Session, entry: m.ModelRegistry, gate: dict[str, Any], promote: bool
) -> None:
    entry.gate = gate
    if promote and gate["passed"]:
        session.execute(
            update(m.ModelRegistry).where(m.ModelRegistry.champion).values(champion=False)
        )
        session.flush()
        entry.champion = True
        entry.promoted_at = datetime.now(UTC)
    session.flush()


def listing(session: Session) -> list[m.ModelRegistry]:
    return list(session.scalars(select(m.ModelRegistry).order_by(m.ModelRegistry.created_at)))


def serving_run(models_dir: Path) -> Path:
    """The champion's run folder, or the latest run if there is no champion or database."""
    try:
        from petlens.db.session import session_scope

        with session_scope() as session:
            best = champion(session)
            run_id = best.run_id if best else None
    except Exception as exc:  # no database: serve the latest run
        log.warning("model registry unavailable (%s); serving the latest run", type(exc).__name__)
        run_id = None
    if run_id:
        for path in all_runs(models_dir):
            if path.name == run_id:
                return path
        log.warning("champion %s not found in %s; serving the latest run", run_id, models_dir)
    return latest_run(models_dir)
