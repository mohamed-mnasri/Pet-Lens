"""Command line entry point: ``python -m petlens.models <stage>``.

Stages: embed, screen, train, compare, the registry ones (Step 11): golden (freeze the
golden set), register, gate (compare with the champion) and promote, and the Step 12
ablations: finetune (tune the last blocks, then embed and train heads on top) and
ablation (compare runs on the latest dataset version through the promotion gate).
``--model`` on embed and train picks another frozen backbone (ViT-B/14).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from petlens.config import Settings, get_settings
from petlens.data.catalog import REPO_DATA_DIR, load_breeds
from petlens.data.manifest import ManifestRow, read_manifest
from petlens.models import embed, golden, policy, screening, train


def admitted_for(version: str) -> frozenset[str]:
    path = REPO_DATA_DIR / "screening" / f"{version}.parquet"
    return screening.admitted(screening.read(path)) if path.exists() else frozenset()


def train_and_save(
    settings: Settings,
    version: str,
    rows: list[ManifestRow],
    model: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Train heads on ``model``'s embeddings of this version and save the run."""
    path = embed.cache_path(settings.data_dir, model).with_name(f"{version}.parquet")
    if not path.exists():
        raise SystemExit(f"no {model} embeddings for this dataset version; run `make embed`")
    vecs = embed.read_embeddings(path)
    policy_path = REPO_DATA_DIR / "policy.yaml"
    base = policy.load(policy_path) if policy_path.exists() else policy.PolicyConfig()
    gold = golden.ids()
    kept = [r for r in rows if not (r.image_id in gold and r.split in train.POOL)]
    if len(kept) < len(rows):
        logging.warning("left %d golden photos out of training", len(rows) - len(kept))
    heads, metrics = train.train(
        kept, vecs, load_breeds(), admitted=admitted_for(version), base_policy=base
    )
    if extra:
        metrics.update(extra)
    if model == embed.DEFAULT_MODEL:  # experiments don't rewrite the committed thresholds
        tuned = policy.PolicyConfig.model_validate(metrics["policy_config"])
        tuned.tuned_on = f"{tuned.tuned_on}; dataset {version}"
        policy.save(tuned, policy_path)
    return train.save_run(settings.models_dir, version, model, heads, metrics)


def embed_version(
    settings: Settings,
    version: str,
    rows: list[ManifestRow],
    model: str,
    batch_size: int = 64,
    device: str | None = None,
) -> Path:
    fn = embed.load_dinov2(model, device, settings.models_dir)
    vecs = embed.embed_rows(rows, settings.data_dir, fn, model, batch_size)
    out = embed.cache_path(settings.data_dir, model).with_name(f"{version}.parquet")
    embed.write_embeddings(vecs, out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.models")
    sub = parser.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("embed", help="DINOv2 embeddings for the latest dataset version")
    e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    e.add_argument("--model", default=embed.DEFAULT_MODEL)
    sub.add_parser("screen", help="screen Commons breed photos with the v0 head")
    t = sub.add_parser("train", help="train and evaluate the heads")
    t.add_argument("--model", default=embed.DEFAULT_MODEL)
    f = sub.add_parser("finetune", help="tune the backbone's last blocks, then train heads")
    f.add_argument("--base", default=embed.DEFAULT_MODEL)
    f.add_argument("--blocks", type=int, default=2, help="blocks to unfreeze")
    f.add_argument("--epochs", type=int, default=6)
    f.add_argument("--batch-size", type=int, default=32)
    f.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("ablation", help="compare runs of the latest dataset version")
    a.add_argument("--reference", default=None, help="run to compare against (default: frozen)")
    sub.add_parser("compare", help="look-alike breeds in the last two training runs")
    g = sub.add_parser("golden", help="freeze the golden set (once) or show it")
    g.add_argument("--per-breed", type=int, default=10)
    sub.add_parser("register", help="record the latest run in the model registry")
    for name in ("gate", "promote"):
        p = sub.add_parser(name, help=f"{name}: compare a run with the champion")
        p.add_argument("--run", default=None, help="run id or prefix (default: the latest)")
    sub.add_parser("registry", help="list registered runs")
    args = parser.parse_args(argv)
    if args.stage == "compare":
        print(train.compare_runs(get_settings().models_dir))
        return
    if args.stage in ("register", "gate", "promote", "registry"):
        from petlens.models import cli_registry

        cli_registry.run(args.stage, getattr(args, "run", None))
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    version = (REPO_DATA_DIR / "manifests" / "LATEST").read_text(encoding="utf-8").strip()
    rows = [
        r
        for r in read_manifest(REPO_DATA_DIR / "manifests" / f"{version}.parquet")
        if r.split != "excluded"
    ]
    model = getattr(args, "model", embed.DEFAULT_MODEL)

    if args.stage == "ablation":
        from petlens.models import ablation

        report = ablation.report(settings, version, args.reference)
        out = REPO_DATA_DIR.parent / "docs" / "results" / f"ablation-{version}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(report)
        print(f"written to {out}")
        return

    if args.stage == "finetune":
        from petlens.models import finetune

        cfg = finetune.FinetuneConfig(
            base=args.base,
            unfreeze_blocks=args.blocks,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        name = finetune.model_name(cfg, version)
        info = finetune.finetune(
            rows,
            load_breeds(),
            settings.data_dir,
            settings.models_dir / "finetune" / name,
            cfg,
            admitted=admitted_for(version),
            golden=golden.ids(),
        )
        finetune.reset_cache(settings.data_dir, name)
        embed_version(settings, version, rows, name)
        run = train_and_save(settings, version, rows, name, {"finetune": info})
        print((run / "report.md").read_text(encoding="utf-8"))
        return

    if args.stage == "golden":
        frozen = golden.read()
        if not frozen:
            picked = golden.select(rows, load_breeds(), version, per_breed=args.per_breed)
            golden.write(picked)
            frozen = golden.read()
            print(f"froze {len(frozen)} photos into {golden.GOLDEN_PATH}")
        print(golden.summary(frozen))
        known = {(s, b) for s in ("cat", "dog") for b in load_breeds().training_classes(s)}
        missing = sorted(b for s, b in known - {(g.species, g.breed) for g in frozen})
        if missing:
            print(f"  no golden photos yet: {', '.join(missing)}")
        return

    if args.stage == "embed":
        out = embed_version(settings, version, rows, model, args.batch_size, args.device)
        print(f"embeddings for this version: {out}")
    elif args.stage == "screen":
        path = embed.cache_path(settings.data_dir, model).with_name(f"{version}.parquet")
        if not path.exists():
            raise SystemExit("no embeddings for this dataset version; run `make embed` first")
        screen_path = REPO_DATA_DIR / "screening" / f"{version}.parquet"
        results = screening.screen(rows, embed.read_embeddings(path), load_breeds())
        screening.write(results, screen_path)
        report = screening.render_report(version, results)
        screen_path.with_suffix(".md").write_text(report, encoding="utf-8")
        print(report)
    else:
        run = train_and_save(settings, version, rows, model)
        print((run / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
