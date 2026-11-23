"""``python -m petlens.eval`` (``make eval``): evaluate the latest training run.

Reads the run's heads, its dataset version's manifest and embeddings, and the decision
policy, then writes the metrics to the ``eval_*`` tables and the dataset composition to
``dataset_count``. ``dbt build`` (``make dbt``) turns those into marts.
"""

from __future__ import annotations

import argparse
import json

import joblib

from petlens.config import get_settings
from petlens.data.catalog import REPO_DATA_DIR, load_breeds
from petlens.data.manifest import read_manifest
from petlens.db.session import session_scope
from petlens.eval import compute, store
from petlens.models import embed, policy, screening
from petlens.vision.load import latest_run


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(prog="python -m petlens.eval").parse_args(argv)
    settings = get_settings()
    run = latest_run(settings.models_dir)
    info = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    version, model = info["dataset_version"], info["embedding_model"]
    rows = read_manifest(REPO_DATA_DIR / "manifests" / f"{version}.parquet")
    vecs = embed.read_embeddings(
        embed.cache_path(settings.data_dir, model).with_name(f"{version}.parquet")
    )
    screen_path = REPO_DATA_DIR / "screening" / f"{version}.parquet"
    admitted = (
        screening.admitted(screening.read(screen_path)) if screen_path.exists() else frozenset()
    )
    cfg = policy.load(REPO_DATA_DIR / "policy.yaml")
    heads = joblib.load(run / "heads.joblib")
    result = compute.evaluate(rows, vecs, heads, load_breeds(), cfg, admitted)
    with session_scope() as session:
        eval_id = store.save(
            session, result, info["run_id"], version, model, compute.composition(rows)
        )
    print(f"evaluation {eval_id} of run {info['run_id']} ({version})")
    for mt in result.metrics:
        if mt["metric"] in ("top1", "false_breed_claim_rate", "unknown_rejection_rate"):
            ci = f" [{mt['lo']:.1%}, {mt['hi']:.1%}]" if mt["lo"] is not None else ""
            print(f"  {mt['species']} {mt['test_set']} {mt['metric']}: {mt['value']:.1%}{ci}")
    print("Next: make dbt")


if __name__ == "__main__":
    main()
