"""Command line entry point: ``python -m petlens.models <embed|train>``."""

from __future__ import annotations

import argparse
import logging

from petlens.config import get_settings
from petlens.data.catalog import REPO_DATA_DIR, load_breeds
from petlens.data.manifest import read_manifest
from petlens.models import embed, policy, screening, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.models")
    sub = parser.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("embed", help="DINOv2 embeddings for the latest dataset version")
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    sub.add_parser("screen", help="screen Commons breed photos with the v0 head")
    sub.add_parser("train", help="train and evaluate the heads")
    sub.add_parser("compare", help="look-alike breeds in the last two training runs")
    args = parser.parse_args(argv)
    if args.stage == "compare":
        print(train.compare_runs(get_settings().models_dir))
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    version = (REPO_DATA_DIR / "manifests" / "LATEST").read_text(encoding="utf-8").strip()
    rows = [
        r
        for r in read_manifest(REPO_DATA_DIR / "manifests" / f"{version}.parquet")
        if r.split != "excluded"
    ]
    model = embed.DEFAULT_MODEL

    if args.stage == "embed":
        fn = embed.load_dinov2(model, args.device)
        vecs = embed.embed_rows(rows, settings.data_dir, fn, model, args.batch_size)
        out = embed.cache_path(settings.data_dir, model).with_name(f"{version}.parquet")
        embed.write_embeddings(vecs, out)
        print(f"embeddings for {len(vecs)} images: {out}")
    else:
        path = embed.cache_path(settings.data_dir, model).with_name(f"{version}.parquet")
        if not path.exists():
            raise SystemExit("no embeddings for this dataset version; run `make embed` first")
        vecs = embed.read_embeddings(path)
        screen_path = REPO_DATA_DIR / "screening" / f"{version}.parquet"
        if args.stage == "screen":
            results = screening.screen(rows, vecs, load_breeds())
            screening.write(results, screen_path)
            report = screening.render_report(version, results)
            screen_path.with_suffix(".md").write_text(report, encoding="utf-8")
            print(report)
            return
        admitted = (
            screening.admitted(screening.read(screen_path)) if screen_path.exists() else frozenset()
        )
        policy_path = REPO_DATA_DIR / "policy.yaml"
        base = policy.load(policy_path) if policy_path.exists() else policy.PolicyConfig()
        heads, metrics = train.train(rows, vecs, load_breeds(), admitted=admitted, base_policy=base)
        tuned = policy.PolicyConfig.model_validate(metrics["policy_config"])
        tuned.tuned_on = f"{tuned.tuned_on}; dataset {version}"
        policy.save(tuned, policy_path)
        run = train.save_run(settings.models_dir, version, model, heads, metrics)
        print((run / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
