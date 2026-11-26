"""The breed-naming experiment (Step 14): can the local VLM name the breed zero-shot?

A one-off script, not a pipeline stage. On the golden set and the unseen-source test,
the VLM picks a breed for each photo from the same list of breeds the classifier knows
(plus "domestic, no clear breed" for cats), with the answer constrained to that list by
a JSON schema. The served classifier (the registry's champion) answers the same photos,
leaving out anything it trained on. The report gives both accuracies with 95% intervals,
the paired split (both right, only one right, both wrong), and the time per photo. It
goes in the README whichever way it lands.

Answers are saved as they come (``<MODELS_DIR>/vlm_breeds/<model>-<prompt>.jsonl``), so
an interrupted run resumes.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from petlens.config import Settings
from petlens.data.catalog import BreedCatalog, Species, load_breeds
from petlens.data.imaging import stored_path
from petlens.data.manifest import ManifestRow
from petlens.eval.compute import wilson
from petlens.models import golden
from petlens.models.promotion import Judge, Photo
from petlens.models.registry import serving_run
from petlens.models.runs import cached_vectors, load_run
from petlens.vlm.screen import OllamaVlm, Vlm, to_jpeg

PROMPT_VERSION = "breeds-v1"
PROMPT = """This photo shows a {species}. Which breed is it? Pick exactly one answer from
this list (the id before the colon), using the coat, colour pattern, face and body:
{choices}
Answer only with JSON: {{"breed": "<id>"}}."""


class BreedAnswer(BaseModel):
    breed: str


def choices(breeds: BreedCatalog, species: Species) -> list[tuple[str, str]]:
    out = []
    for b in breeds.training_classes(species):
        name = "domestic cat, no clear breed" if b == "domestic_cat" else breeds.get(b).name
        out.append((b, name))
    return out


def ask(vlm: Vlm, image_jpeg: bytes, species: str, options: list[tuple[str, str]]) -> str | None:
    """The VLM's pick (an id from ``options``), one retry, else None."""
    ids = [i for i, _ in options]
    prompt = PROMPT.format(
        species=species, choices="\n".join(f"- {i}: {name}" for i, name in options)
    )
    schema = {
        "type": "object",
        "properties": {"breed": {"type": "string", "enum": ids}},
        "required": ["breed"],
    }
    for _ in (1, 2):
        try:
            raw = vlm.ask(prompt, image_jpeg, schema)
        except httpx.TimeoutException:  # a photo that sends the model into a loop
            continue
        try:
            pick = BreedAnswer.model_validate(json.loads(raw.strip())).breed
        except (ValueError, ValidationError):
            continue
        if pick in ids:
            return pick
    return None


@dataclass
class Item:
    test_set: str
    photo: Photo


def photos(rows: list[ManifestRow], breeds: BreedCatalog, limit: int | None) -> list[Item]:
    known = {s: set(breeds.training_classes(s)) for s in ("cat", "dog")}
    gold = [Item("golden", Photo(g.image_id, g.species, g.breed)) for g in golden.read()]
    gold_ids = {i.photo.image_id for i in gold}
    unseen = [
        Item("test_unseen_source", Photo(r.image_id, r.species, r.breed))
        for r in rows
        if r.split == "test_unseen_source"
        and r.breed in known.get(r.species, set())
        and r.image_id not in gold_ids
    ]

    def pick(items: list[Item]) -> list[Item]:
        ordered = sorted(items, key=lambda i: hashlib.sha256(i.photo.image_id.encode()).hexdigest())
        return ordered[:limit] if limit else ordered

    return pick(gold) + pick(unseen)


def _cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["image_id"]] = d
    return out


def summarise(results: list[dict[str, Any]], model: str, classifier: str) -> str:
    lines = [
        f"# Breed naming: {model} zero-shot vs the classifier",
        "",
        f"Classifier: {classifier}. The VLM picks from the same breed list (prompt "
        f"{PROMPT_VERSION}). Photos the classifier trained on are left out.",
        "",
        "| Test set | Photos | VLM top-1 | Classifier top-1 | Both right | Only VLM | "
        "Only classifier | Both wrong | VLM no answer | VLM s/photo |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for test in ("golden", "test_unseen_source"):
        rs = [r for r in results if r["test_set"] == test]
        if not rs:
            continue
        n = len(rs)
        v = [r["vlm"] == r["truth"] for r in rs]
        c = [r["classifier"] == r["truth"] for r in rs]

        def acc(xs: list[bool]) -> str:
            lo, hi = wilson(sum(xs), len(xs))
            return f"{sum(xs) / len(xs):.1%} ({lo:.0%} to {hi:.0%})"

        secs = sorted(r["seconds"] for r in rs)
        lines.append(
            f"| {test} | {n} | {acc(v)} | {acc(c)} "
            f"| {sum(a and b for a, b in zip(v, c, strict=True))} "
            f"| {sum(a and not b for a, b in zip(v, c, strict=True))} "
            f"| {sum(b and not a for a, b in zip(v, c, strict=True))} "
            f"| {sum(not a and not b for a, b in zip(v, c, strict=True))} "
            f"| {sum(r['vlm'] is None for r in rs)} | {secs[len(secs) // 2]:.1f} |"
        )
    wrong = [r for r in results if r["vlm"] and r["vlm"] != r["truth"]]
    common: dict[tuple[str, str], int] = {}
    for r in wrong:
        common[(r["truth"], r["vlm"])] = common.get((r["truth"], r["vlm"]), 0) + 1
    if common:
        top = sorted(common.items(), key=lambda kv: -kv[1])[:8]
        lines += [
            "",
            "VLM's most common mistakes: "
            + ", ".join(f"{t} called {p} ({k})" for (t, p), k in top),
        ]
    return "\n".join(lines) + "\n"


def run_experiment(
    settings: Settings, rows: list[ManifestRow], limit: int | None, vlm: Vlm | None = None
) -> str:
    breeds = load_breeds()
    vlm = vlm or OllamaVlm(settings.vlm_model, settings.ollama_url)
    run = load_run(serving_run(settings.models_dir))
    judge = Judge(run, cached_vectors(run.embedding_model, settings.data_dir), [])
    items = [i for i in photos(rows, breeds, limit) if judge.usable(i.photo)]
    top1, _ = judge.answers([i.photo for i in items])
    cache_path = (
        settings.models_dir / "vlm_breeds" / f"{vlm.model.replace(':', '_')}-{PROMPT_VERSION}.jsonl"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _cache(cache_path)
    images = settings.data_dir / "processed" / "images"
    options = {"cat": choices(breeds, "cat"), "dog": choices(breeds, "dog")}
    results = []
    with cache_path.open("a", encoding="utf-8") as fh:
        for item, cls in zip(items, top1, strict=True):
            p = item.photo
            hit = cache.get(p.image_id)
            if hit is None:
                path = stored_path(images, p.image_id)
                if not path.exists():
                    continue
                started = time.monotonic()
                pick = ask(vlm, to_jpeg(path), p.species, options[p.species])
                hit = {"image_id": p.image_id, "vlm": pick, "seconds": time.monotonic() - started}
                fh.write(json.dumps(hit) + "\n")
                fh.flush()
            results.append(
                {
                    "test_set": item.test_set,
                    "truth": p.breed,
                    "vlm": hit["vlm"],
                    "classifier": cls,
                    "seconds": hit["seconds"],
                }
            )
    return summarise(results, vlm.model, run.run_id)
