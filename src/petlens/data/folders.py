"""Import a folder that holds one subfolder of photos per breed.

``make import-folders FOLDER="C:\\Users\\me\\Downloads\\cats" LICENCE="Pexels licence"``

Subfolder names are matched to breeds loosely ("Maine Coon", "maine-coon",
"Maine Coon cats" all work). Each photo goes through the same checks as the page's
"Add training photos" tab (``petlens.data.own``) and is listed in the ``licensed``
source with the licence I give, so its lineage stays on every row. Rerunning skips
photos already imported.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from petlens.data.catalog import BreedCatalog
from petlens.data.imaging import sha256_bytes
from petlens.data.own import add_photo, known_refs
from petlens.vision.detect import Detector

log = logging.getLogger(__name__)

SOURCE_KEY = "licensed"
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _norm(text: str) -> str:
    words = re.sub(r"[^a-z]+", " ", text.lower()).split()
    while words and words[-1] in ("cat", "cats", "dog", "dogs", "breed", "photos", "images"):
        words.pop()
    return " ".join(words)


def match_breed(folder_name: str, breeds: BreedCatalog) -> str | None:
    """The breed id a folder name means, or ``None``."""
    want = _norm(folder_name)
    for b in breeds.breeds:
        names = {_norm(b.id), _norm(b.name)}
        if want in names or want.rstrip("s") in names:
            return b.id
    return None


@dataclass
class FolderReport:
    added: Counter[str] = field(default_factory=Counter)
    already: Counter[str] = field(default_factory=Counter)
    left_out: Counter[str] = field(default_factory=Counter)
    reasons: Counter[str] = field(default_factory=Counter)
    unmatched: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = ["breed: added / already there / left out"]
        for breed in sorted({*self.added, *self.already, *self.left_out}):
            lines.append(
                f"  {breed}: {self.added[breed]} / {self.already[breed]} / {self.left_out[breed]}"
            )
        if self.reasons:
            lines.append("why photos were left out:")
            lines += [f"  {why}: {n}" for why, n in self.reasons.most_common()]
        for name in self.unmatched:
            lines.append(f"skipped folder '{name}': no breed by that name (rename it?)")
        return "\n".join(lines)


def import_folders(
    root: Path,
    licence: str,
    data_dir: Path,
    breeds: BreedCatalog,
    detect: Detector,
    attribution: str = "",
) -> FolderReport:
    if not root.is_dir():
        raise ValueError(f"folder not found: {root}")
    report = FolderReport()
    known = known_refs(data_dir, SOURCE_KEY)
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        breed = match_breed(sub.name, breeds)
        if breed is None or breeds.get(breed).role != "train":
            report.unmatched.append(sub.name)
            continue
        for file in sorted(sub.iterdir()):
            if file.suffix.lower() not in PHOTO_SUFFIXES:
                continue
            data = file.read_bytes()
            if sha256_bytes(data) in known:
                report.already[breed] += 1
                continue
            r = add_photo(
                data,
                breed,
                data_dir,
                breeds,
                detect,
                licence=licence,
                attribution=attribution or licence,
                source=SOURCE_KEY,
            )
            if r.status == "added":
                report.added[breed] += 1
            elif r.status == "already":
                report.already[breed] += 1
            else:
                report.left_out[breed] += 1
                report.reasons[r.reason or "?"] += 1
        log.info("%s done", sub.name)
    return report


def main(argv: list[str] | None = None) -> None:
    from petlens.config import get_settings
    from petlens.data.catalog import load_breeds
    from petlens.vision.detect import DetectorConfig, load_rfdetr

    parser = argparse.ArgumentParser(prog="python -m petlens.data.folders")
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--licence", required=True, help='e.g. "Pexels licence" or "CC0"')
    parser.add_argument("--attribution", default="")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    detect = load_rfdetr(DetectorConfig(size=settings.detector_size))
    report = import_folders(
        args.folder, args.licence, settings.data_dir, load_breeds(), detect, args.attribution
    )
    print(report.render())
    print("Next: make data-build, make embed, make screen, make train.")


if __name__ == "__main__":
    main()
