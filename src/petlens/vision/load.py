"""Build an ``Analyzer`` from the served run (the registry's champion, else the latest),
its dataset version and its own decision policy."""

from __future__ import annotations

import numpy as np

from petlens.config import Settings
from petlens.data.catalog import load_breeds
from petlens.models import embed
from petlens.models.openset import KnnIndex
from petlens.models.registry import serving_run
from petlens.models.runs import latest_run, load_run, version_vectors
from petlens.vision.detect import DetectorConfig, load_rfdetr
from petlens.vision.pipeline import Analyzer

__all__ = ["latest_run", "load_analyzer"]


def load_analyzer(settings: Settings, detector_size: str = "base") -> Analyzer:
    run = load_run(serving_run(settings.models_dir))
    vecs = version_vectors(run, settings.data_dir)
    knn = {}
    for species in run.heads:
        train_x = [vecs[i] for i in run.training_ids(species) if i in vecs]
        knn[species] = KnnIndex(np.stack(train_x).astype(np.float64))
    breeds = load_breeds()
    return Analyzer(
        detector=load_rfdetr(DetectorConfig(size=detector_size)),
        embed=embed.load_dinov2(run.embedding_model),
        heads=run.heads,
        knn=knn,
        policy=run.cfg,
        close_pairs=[(a, b) for a, b in breeds.close_pairs],
        version=run.path.name,
    )
