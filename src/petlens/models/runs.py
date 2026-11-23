"""Load a saved training run: its heads, decision policy, dataset rows and training pool.

A run folder (``<MODELS_DIR>/runs/<run_id>``) holds ``heads.joblib`` and
``metrics.json``. The rest comes from the repo (the manifest and screening results of
the run's dataset version) and the embedding cache of the run's embedding model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from petlens.data.catalog import REPO_DATA_DIR
from petlens.data.manifest import ManifestRow, read_manifest
from petlens.models import embed, policy, screening
from petlens.models.heads import SpeciesHead
from petlens.models.train import in_training_pool

Vectors = dict[str, NDArray[np.float32]]


def all_runs(models_dir: Path) -> list[Path]:
    return sorted(p for p in (models_dir / "runs").glob("*") if (p / "heads.joblib").exists())


def latest_run(models_dir: Path) -> Path:
    runs = all_runs(models_dir)
    if not runs:
        raise SystemExit("no trained run found; run `make train` first")
    return runs[-1]


def find_run(models_dir: Path, run_id: str | None) -> Path:
    """A run by id (or a unique prefix of it); the latest when ``run_id`` is None."""
    if run_id is None:
        return latest_run(models_dir)
    matches = [p for p in all_runs(models_dir) if p.name.startswith(run_id)]
    if len(matches) != 1:
        raise SystemExit(f"{len(matches)} runs match {run_id!r}")
    return matches[0]


@dataclass
class LoadedRun:
    path: Path
    info: dict[str, Any]
    heads: dict[str, SpeciesHead]
    cfg: policy.PolicyConfig
    rows: list[ManifestRow] = field(repr=False)
    admitted: frozenset[str] = field(repr=False)

    @property
    def run_id(self) -> str:
        return str(self.info["run_id"])

    @property
    def dataset_version(self) -> str:
        return str(self.info["dataset_version"])

    @property
    def embedding_model(self) -> str:
        return str(self.info["embedding_model"])

    def training_ids(self, species: str) -> list[str]:
        return [
            r.image_id
            for r in self.rows
            if r.species == species and r.split == "train" and in_training_pool(r, self.admitted)
        ]


def run_policy(info: dict[str, Any], repo_data: Path = REPO_DATA_DIR) -> policy.PolicyConfig:
    """The thresholds the run was tuned with. Older runs fall back to ``policy.yaml``."""
    stored = info.get("metrics", {}).get("policy_config")
    if stored:
        return policy.PolicyConfig.model_validate(stored)
    path = repo_data / "policy.yaml"
    return policy.load(path) if path.exists() else policy.PolicyConfig()


def load_run(path: Path, repo_data: Path = REPO_DATA_DIR) -> LoadedRun:
    info = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    version = info["dataset_version"]
    rows = read_manifest(repo_data / "manifests" / f"{version}.parquet")
    screen_path = repo_data / "screening" / f"{version}.parquet"
    admitted = (
        screening.admitted(screening.read(screen_path)) if screen_path.exists() else frozenset()
    )
    return LoadedRun(
        path=path,
        info=info,
        heads=joblib.load(path / "heads.joblib"),
        cfg=run_policy(info, repo_data),
        rows=rows,
        admitted=admitted,
    )


def version_vectors(run: LoadedRun, data_dir: Path) -> Vectors:
    """Embeddings of the run's dataset version (written by ``make embed``)."""
    return embed.read_embeddings(
        embed.cache_path(data_dir, run.embedding_model).with_name(f"{run.dataset_version}.parquet")
    )


def cached_vectors(model_name: str, data_dir: Path) -> Vectors:
    """Every image ever embedded with ``model_name``, across dataset versions."""
    return embed.read_embeddings(embed.cache_path(data_dir, model_name))
