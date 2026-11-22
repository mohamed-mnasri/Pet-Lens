"""Build an ``Analyzer`` from the latest dataset version, training run and policy."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from petlens.config import Settings
from petlens.data.catalog import REPO_DATA_DIR, load_breeds
from petlens.data.manifest import read_manifest
from petlens.models import embed, policy, screening
from petlens.models.openset import KnnIndex
from petlens.models.train import in_training_pool
from petlens.vision.detect import DetectorConfig, load_rfdetr
from petlens.vision.pipeline import Analyzer


def latest_run(models_dir: Path) -> Path:
    runs = sorted(p for p in (models_dir / "runs").glob("*") if (p / "heads.joblib").exists())
    if not runs:
        raise SystemExit("no trained run found; run `make train` first")
    return runs[-1]


def load_analyzer(settings: Settings, detector_size: str = "base") -> Analyzer:
    run = latest_run(settings.models_dir)
    info = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    version = info["dataset_version"]
    model_name = info["embedding_model"]
    heads = joblib.load(run / "heads.joblib")

    rows = read_manifest(REPO_DATA_DIR / "manifests" / f"{version}.parquet")
    screen_path = REPO_DATA_DIR / "screening" / f"{version}.parquet"
    admitted = (
        screening.admitted(screening.read(screen_path)) if screen_path.exists() else frozenset()
    )
    vecs = embed.read_embeddings(
        embed.cache_path(settings.data_dir, model_name).with_name(f"{version}.parquet")
    )
    knn = {}
    for species in heads:
        train_x = [
            vecs[r.image_id]
            for r in rows
            if r.species == species
            and r.split == "train"
            and r.image_id in vecs
            and in_training_pool(r, admitted)
        ]
        knn[species] = KnnIndex(np.stack(train_x).astype(np.float64))

    policy_path = REPO_DATA_DIR / "policy.yaml"
    cfg = policy.load(policy_path) if policy_path.exists() else policy.PolicyConfig()
    breeds = load_breeds()
    return Analyzer(
        detector=load_rfdetr(DetectorConfig(size=detector_size)),
        embed=embed.load_dinov2(model_name),
        heads=heads,
        knn=knn,
        policy=cfg,
        close_pairs=[(a, b) for a, b in breeds.close_pairs],
        version=run.name,
    )
