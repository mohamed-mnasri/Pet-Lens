"""Analyse a photo or a short clip: ``python -m petlens.vision <file> [<file> ...]``."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from PIL import Image

from petlens.config import get_settings
from petlens.models.policy import Decision
from petlens.vision.load import load_analyzer
from petlens.vision.pipeline import analyse_clip, analyse_image
from petlens.vision.video import sample_frames

VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".m4v"}


def describe(species: str, d: Decision) -> str:
    top3 = ", ".join(f"{name} {p:.0%}" for name, p in d.top3)
    if d.outcome == "accept":
        name = f"{d.label} (or {d.also})" if d.also else d.label
        head = f"looks like a {name}, {d.confidence:.0%}"
    elif d.outcome == "no_clear_breed":
        head = "domestic cat, no clear breed" if species == "cat" else "no clear breed"
    else:
        head = "not sure, sent to review"
    return f"{species}: {head}  [top 3: {top3}; kNN distance {d.knn_distance:.3f}]"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m petlens.vision")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--detector", default="base", help="RF-DETR size: nano/small/medium/base")
    parser.add_argument("--save-blurred", type=Path, help="folder for the people-blurred copies")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    t0 = time.perf_counter()
    analyzer = load_analyzer(get_settings(), args.detector)
    print(f"models loaded in {time.perf_counter() - t0:.1f} s")

    for path in args.files:
        print(f"\n== {path.name}")
        if path.suffix.lower() in VIDEO_SUFFIXES:
            frames = sample_frames(path)
            clip = analyse_clip(analyzer, frames)
            per_frame = clip.seconds / max(len(frames), 1)
            print(f"{len(frames)} frames, {clip.seconds:.2f} s ({per_frame:.2f} s per frame)")
            if clip.ambiguous:
                print(f"grouping is ambiguous, per-frame results instead: {clip.reason}")
                for f, res in zip(frames, clip.frames, strict=True):
                    for a in res.animals:
                        what = describe(a.species, a.decision)
                        print(f"  frame {f.index} ({f.seconds:.0f} s): {what}")
            else:
                print(f"{len(clip.groups)} animal(s)")
                for g in clip.groups:
                    print(f"  frames {g.frames}: {describe(g.species, g.decision)}")
        else:
            with Image.open(path) as im:
                res = analyse_image(analyzer, im)
            n_animals, n_people = len(res.animals), res.people_blurred
            print(f"{res.seconds:.2f} s, {n_animals} animal(s), {n_people} person(s) blurred")
            for a in res.animals:
                print(f"  {describe(a.species, a.decision)}")
            if args.save_blurred and res.people_blurred:
                args.save_blurred.mkdir(parents=True, exist_ok=True)
                res.image.save(args.save_blurred / f"{path.stem}.blurred.jpg", quality=90)


if __name__ == "__main__":
    main()
