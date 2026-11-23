"""The Step 12 ablation report: every backbone tried on one dataset version, side by side.

For each embedding model (frozen ViT-S/14, ViT-S/14 with its last blocks tuned, frozen
ViT-B/14, ViT-B/14 tuned) the latest run of the dataset version is shown with its test
numbers, its training cost, and the promotion gate against the reference run (frozen
ViT-S by default). The winner is picked by the gate, not by in-domain accuracy: the best
golden-set gain among runs that pass every check. If none passes, the reference stays.
"""

from __future__ import annotations

import json
from typing import Any

from petlens.config import Settings
from petlens.data.catalog import load_breeds
from petlens.models import embed, golden, promotion
from petlens.models.runs import LoadedRun, all_runs, find_run, load_run


def _pct(v: float | None) -> str:
    return "" if v is None else f"{v:.1%}"


def latest_per_model(settings: Settings, version: str) -> dict[str, LoadedRun]:
    out: dict[str, LoadedRun] = {}
    for path in all_runs(settings.models_dir):  # sorted oldest first, so the latest wins
        info = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        if info["dataset_version"] == version:
            out[info["embedding_model"]] = load_run(path)
    return out


def _numbers(run: LoadedRun) -> dict[str, Any]:
    m = run.info["metrics"]
    cat, dog = m.get("cat", {}), m.get("dog", {})

    def top1(species: dict[str, Any], test: str) -> float | None:
        t = species.get("tests", {}).get(test)
        return t["calibrated"]["top1"] if t else None

    pol = cat.get("policy", {})
    ft = m.get("finetune") or {}
    return {
        "cat_in": top1(cat, "test_in_domain"),
        "cat_unseen": top1(cat, "test_unseen_source"),
        "dog_in": top1(dog, "test_in_domain"),
        "fbc": pol.get("test_domestic", {}).get("false_breed_claim_rate"),
        "unknown": pol.get("test_unknown", {}).get("unknown_rejection_rate"),
        "minutes": round(ft["seconds"] / 60, 1) if ft.get("seconds") else None,
        "gpu_mb": ft.get("peak_gpu_mb"),
    }


def report(settings: Settings, version: str, reference: str | None = None) -> str:
    runs = latest_per_model(settings, version)
    if not runs:
        return f"no runs for {version} yet; run `make train` first"
    if reference:
        ref = load_run(find_run(settings.models_dir, reference))
    elif embed.DEFAULT_MODEL in runs:
        ref = runs[embed.DEFAULT_MODEL]
    else:
        return "no frozen ViT-S run for this version; run `make train` or pass --reference"
    gold, breeds = golden.read(), load_breeds()
    lines = [
        f"# Backbone ablation on {version}",
        "",
        f"Reference: {ref.embedding_model} ({ref.run_id}). Same splits, seed and training "
        "pool for every run. Gate: paired bootstrap on the golden set and the unseen-source "
        "test, per-breed recall margin, false breed claim rate.",
        "",
        "| Backbone | Cat Oxford | Cat unseen | Dog Oxford | False claims | Unknown rejected "
        "| Tuning time | GPU peak | Golden gain | Unseen gain | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    winner: tuple[float, str] | None = None
    failures: list[str] = []
    for name, run in sorted(runs.items()):
        n = _numbers(run)
        if run.run_id == ref.run_id:
            gain_g = gain_u = "reference"
            verdict = ""
        else:
            result = promotion.compare(ref, run, gold, settings.data_dir, breeds)
            g = result.numbers.get("golden set", {})
            u = result.numbers.get("unseen-source test", {})
            gain_g = f"{g['diff']:+.1%}" if g else ""
            gain_u = f"{u['diff']:+.1%}" if u else ""
            verdict = "passes" if result.passed else "fails"
            if result.passed and g and (winner is None or g["diff"] > winner[0]):
                winner = (g["diff"], name)
            failures += [f"- {name}: {c.name}: {c.detail}" for c in result.checks if not c.passed]
        minutes = "" if n["minutes"] is None else f"{n['minutes']} min"
        gpu = "" if n["gpu_mb"] is None else f"{n['gpu_mb']} MB"
        lines.append(
            f"| {name} | {_pct(n['cat_in'])} | {_pct(n['cat_unseen'])} | {_pct(n['dog_in'])} "
            f"| {_pct(n['fbc'])} | {_pct(n['unknown'])} | {minutes} | {gpu} "
            f"| {gain_g} | {gain_u} | {verdict} |"
        )
    lines.append("")
    if winner:
        lines.append(f"**Winner: {winner[1]}** (golden-set gain {winner[0]:+.1%}).")
    else:
        lines.append(f"**No run beat the reference under the gate: {ref.embedding_model} stays.**")
    if failures:
        lines += ["", "Checks that failed:", "", *failures]
    return "\n".join(lines) + "\n"
