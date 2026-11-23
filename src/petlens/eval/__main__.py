"""``python -m petlens.eval`` (``make eval``): evaluate the latest training run.

Reads the run's heads, its dataset version's manifest and embeddings, and the decision
policy, then writes the metrics to the ``eval_*`` tables and the dataset composition to
``dataset_count``. ``dbt build`` (``make dbt``) turns those into marts.
"""

from __future__ import annotations

import argparse

from petlens.config import get_settings
from petlens.data.catalog import load_breeds
from petlens.db.session import session_scope
from petlens.eval import compute, store
from petlens.models import golden
from petlens.models.runs import cached_vectors, find_run, load_run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.eval")
    parser.add_argument("--run", default=None, help="run id or prefix (default: the latest)")
    args = parser.parse_args(argv)
    settings = get_settings()
    run = load_run(find_run(settings.models_dir, args.run))
    info, version, model = run.info, run.dataset_version, run.embedding_model
    rows = run.rows
    vecs = cached_vectors(model, settings.data_dir)
    result = compute.evaluate(
        rows, vecs, run.heads, load_breeds(), run.cfg, run.admitted, golden=golden.read()
    )
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
