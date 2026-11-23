"""How far to trust the VLM screener (Step 13): agreement and effect, as a report.

- **Agreement with my swipe answers** (Cohen's kappa), on photos I reviewed and the VLM
  tagged. "Not a cat or dog" is compared with ``kind != photo``; "not sure" (which I
  used for poor photos) with the VLM calling the photo unusable (not a real photo, only
  part of the animal, blurry or several animals).
- **Agreement with the pixel and detector checks** from ``make data-quality``: blurry
  against the Laplacian check, whole animal against the box (not cut off, an animal
  found).
- **Effect on training-set noise**: among web photos I reviewed, how often the label
  was wrong (I fixed it or said it isn't a cat or dog) in the photos the VLM keeps
  versus the ones it would drop, and how many training candidates it drops.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from petlens.data.manifest import ManifestRow
from petlens.data.vlm_flags import BLOCKING, SCREENED_SOURCES, flags_from
from petlens.db import models as m
from petlens.vlm.screen import PROMPT_VERSION


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Agreement beyond chance between two yes/no raters (None with no variation)."""
    if not a:
        return None
    x, y = np.array(a, dtype=bool), np.array(b, dtype=bool)
    observed = float((x == y).mean())
    px, py = float(x.mean()), float(y.mean())
    expected = px * py + (1 - px) * (1 - py)
    if expected >= 1:
        return None
    return (observed - expected) / (1 - expected)


@dataclass
class Agreement:
    name: str
    n: int
    kappa: float | None
    both_yes: int
    vlm_only: int
    other_only: int

    def line(self) -> str:
        k = "n/a" if self.kappa is None else f"{self.kappa:.2f}"
        return (
            f"| {self.name} | {self.n} | {k} | {self.both_yes} | {self.vlm_only} | "
            f"{self.other_only} |"
        )


def agreement(name: str, pairs: list[tuple[bool, bool]]) -> Agreement:
    vlm = [p[0] for p in pairs]
    other = [p[1] for p in pairs]
    return Agreement(
        name,
        len(pairs),
        cohen_kappa(vlm, other),
        sum(1 for v, o in pairs if v and o),
        sum(1 for v, o in pairs if v and not o),
        sum(1 for v, o in pairs if o and not v),
    )


def unusable(tag: dict[str, Any]) -> bool:
    return bool(flags_from(tag) - {"vlm_kitten"})


def last_answers(session: Session) -> dict[str, str]:
    """My latest real answer per photo (skips ignored)."""
    q = (
        select(m.LabelEvent.image_id, m.LabelEvent.action)
        .where(
            m.LabelEvent.origin == "me",
            m.LabelEvent.image_id.is_not(None),
            m.LabelEvent.action.in_(("accept", "fix", "reject", "unsure")),
        )
        .order_by(m.LabelEvent.created_at)
    )
    return {i: a for i, a in session.execute(q) if i}


def tags_by_image(
    session: Session, model_tag: str
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    q = select(m.VlmTag).where(
        m.VlmTag.model_tag == model_tag, m.VlmTag.prompt_version == PROMPT_VERSION
    )
    tags: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    seconds = []
    for t in session.scalars(q):
        stats[t.status] += 1
        stats["retried"] += int(t.attempts > 1)
        seconds.append(t.seconds)
        if t.status == "ok":
            tags[t.image_id] = {
                "kind": t.kind,
                "whole_animal": t.whole_animal,
                "kitten": t.kitten,
                "blurry": t.blurry,
                "several_animals": t.several_animals,
            }
    if seconds:
        stats["median_ms"] = round(float(np.median(seconds)) * 1000)
    return tags, stats


def report(session: Session, rows: list[ManifestRow], model_tag: str) -> str:
    tags, stats = tags_by_image(session, model_tag)
    answers = last_answers(session)
    by_id = {r.image_id: r for r in rows}
    total = stats["ok"] + stats["parse_failed"]
    lines = [
        f"# VLM screener: {model_tag}, prompt {PROMPT_VERSION}",
        "",
        f"- Photos tagged: {total} (valid JSON: {stats['ok']}, parse_failed: "
        f"{stats['parse_failed']}, needed the retry: {stats['retried']})",
        f"- Median time per photo: {stats.get('median_ms', 0) / 1000:.1f} s",
    ]
    if tags:
        kinds = Counter(t["kind"] for t in tags.values())
        shares = {
            f: sum(1 for t in tags.values() if f in flags_from(t)) / len(tags)
            for f in ("vlm_partial", "vlm_several", "vlm_blurry", "vlm_kitten")
        }
        lines.append(
            "- Kinds: "
            + ", ".join(f"{k} {n}" for k, n in kinds.most_common())
            + "; "
            + ", ".join(f"{f.removeprefix('vlm_')} {v:.0%}" for f, v in shares.items())
        )
    reviewed = [(i, a) for i, a in answers.items() if i in tags]
    swipe = [
        agreement(
            "not a real photo vs my 'not a cat or dog'",
            [(tags[i]["kind"] != "photo", a == "reject") for i, a in reviewed],
        ),
        agreement(
            "unusable vs my 'not sure'",
            [(unusable(tags[i]), a == "unsure") for i, a in reviewed if a != "reject"],
        ),
    ]
    quality: list[tuple[str, list[tuple[bool, bool]]]] = [
        ("blurry vs the pixel check", []),
        ("whole animal vs the box check", []),
    ]
    for i, t in tags.items():
        r = by_id.get(i)
        if r is None:
            continue
        quality[0][1].append((bool(t["blurry"]), "blurry" in r.flags))
        if r.bbox or "no_animal" in r.flags:  # the detector (or Oxford's mask) had a say
            whole_by_box = not ({"cut_off", "no_animal"} & set(r.flags))
            quality[1][1].append((bool(t["whole_animal"]), whole_by_box))
    checks = [agreement(name, pairs) for name, pairs in quality]
    lines += [
        "",
        "Agreement (Cohen's kappa: 1 is perfect, 0 is chance):",
        "",
        "| Comparison | Photos | Kappa | Both say yes | Only the VLM | Only the other |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *(a.line() for a in swipe + checks),
    ]
    web = [(i, a) for i, a in reviewed if by_id.get(i) and by_id[i].source in SCREENED_SOURCES]
    if web:
        kept = [a for i, a in web if not unusable(tags[i])]
        dropped = [a for i, a in web if unusable(tags[i])]

        def wrong(xs: list[str]) -> str:
            judged = [a for a in xs if a != "unsure"]
            if not judged:
                return "n/a"
            return (
                f"{sum(a in ('fix', 'reject') for a in judged) / len(judged):.0%} of {len(judged)}"
            )

        lines += [
            "",
            "Effect on label noise (web photos I reviewed, 'not sure' left out):",
            "",
            f"- Wrong label among photos the VLM keeps: {wrong(kept)}",
            f"- Wrong label among photos the VLM would drop: {wrong(dropped)}",
        ]
    candidates = [
        r
        for r in rows
        if r.source in SCREENED_SOURCES
        and r.split in ("train", "val", "calibration")
        and r.image_id in tags
    ]
    if candidates:
        n_blocked = sum(
            1
            for r in candidates
            if r.label_origin != "me" and BLOCKING & flags_from(tags[r.image_id])
        )
        lines.append(
            f"- Web training candidates it keeps out of training: {n_blocked} of "
            f"{len(candidates)} tagged ({n_blocked / len(candidates):.0%})"
        )
    return "\n".join(lines) + "\n"
