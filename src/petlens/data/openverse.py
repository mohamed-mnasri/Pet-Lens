"""Extra openly licensed photos per breed from Openverse (https://openverse.org/).

``make data-openverse`` (``--per-breed 30`` by default) searches "<breed name> cat" or
"<breed name> dog", keeps results whose licence allows reuse (CC0, public domain,
CC BY, CC BY-SA), skips Wikimedia results (the commons source has those), and runs each
photo through the same checks as my own photos (``petlens.data.own``): one animal of
the right species, not black and white, blurry, cut off or tiny, faces blurred.

The search query is only a weak label, like a Commons category: these photos go
through screening like Commons photos before they are trained on, and disagreements go
to the review queue. Every row keeps its licence, creator and landing page.

Anonymous Openverse use is rate limited, so requests are spaced out and a 429 stops
the run cleanly; rerunning continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from petlens.data.catalog import BreedCatalog
from petlens.data.own import add_photo, known_refs
from petlens.vision.detect import Detector

log = logging.getLogger(__name__)

SOURCE_KEY = "openverse"
API_URL = "https://api.openverse.org/v1/images/"
USER_AGENT = "PetLens/0.1 (personal research project; https://github.com/mohamed-mnasri/petlens)"
LICENCES = {"cc0": "CC0", "pdm": "Public domain", "by": "CC BY", "by-sa": "CC BY-SA"}
PAGE_SIZE = 20  # the most anonymous requests may ask for
MAX_PAGES = 6

Getter = Callable[[str], bytes]


class RateLimitedError(RuntimeError):
    pass


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data: bytes = resp.read()
            return data
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RateLimitedError("Openverse rate limit reached; rerun later") from exc
        raise


def search_url(query: str, page: int) -> str:
    params = {
        "q": query,
        "license": ",".join(LICENCES),
        "page_size": str(PAGE_SIZE),
        "page": str(page),
        "mature": "false",
    }
    return API_URL + "?" + urllib.parse.urlencode(params)


def licence_of(result: dict[str, Any]) -> str | None:
    name = LICENCES.get(str(result.get("license", "")).lower())
    if name is None:
        return None
    version = result.get("license_version")
    return f"{name} {version}" if version and name.startswith("CC BY") else name


def fetch_breed(
    breed: str,
    breeds: BreedCatalog,
    data_dir: Path,
    detect: Detector,
    per_breed: int = 30,
    get: Getter = _http_get,
    min_interval: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Counter[str]:
    """Add up to ``per_breed`` usable photos of ``breed``. Returns outcome counts."""
    entry = breeds.get(breed)
    query = f"{entry.name} {entry.species}"
    known = known_refs(data_dir, SOURCE_KEY)
    have = sum(1 for r in known if r.startswith(f"openverse:{breed}:"))
    out: Counter[str] = Counter()
    for page in range(1, MAX_PAGES + 1):
        if have >= per_breed:
            break
        sleep(min_interval)
        results = json.loads(get(search_url(query, page))).get("results", [])
        if not results:
            break
        for res in results:
            if have >= per_breed:
                break
            ref = f"openverse:{breed}:{res.get('id')}"
            licence = licence_of(res)
            if ref in known:
                continue
            if res.get("source") == "wikimedia" or licence is None or not res.get("url"):
                out["skipped (Wikimedia or licence)"] += 1
                continue
            try:
                sleep(min_interval / 4)
                data = get(str(res["url"]))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                log.info("download failed for %s: %s", res.get("url"), exc)
                out["download failed"] += 1
                continue
            creator = res.get("creator") or "unknown"
            r = add_photo(
                data,
                breed,
                data_dir,
                breeds,
                detect,
                licence=licence,
                attribution=f"{res.get('title') or 'Untitled'} by {creator}, {licence}, "
                f"{res.get('foreign_landing_url') or res['url']}",
                source=SOURCE_KEY,
                label_origin="openverse_search",
                ref=ref,
                author=creator,
            )
            known.add(ref)
            out[r.status if r.status != "left_out" else f"left out: {r.reason}"] += 1
            if r.status == "added":
                have += 1
    return out


def main(argv: list[str] | None = None) -> None:
    from petlens.config import get_settings
    from petlens.data.catalog import load_breeds
    from petlens.vision.detect import DetectorConfig, load_rfdetr

    parser = argparse.ArgumentParser(prog="python -m petlens.data.openverse")
    parser.add_argument("--per-breed", type=int, default=30)
    parser.add_argument("--species", choices=["cat", "dog", "all"], default="all")
    parser.add_argument("--breed", action="append", help="only these breeds (repeatable)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = get_settings()
    breeds = load_breeds()
    todo = args.breed or [b for b in breeds.training_classes(args.species) if b != "domestic_cat"]
    detect = load_rfdetr(DetectorConfig(size=settings.detector_size))
    try:
        for breed in todo:
            got = fetch_breed(breed, breeds, settings.data_dir, detect, args.per_breed)
            summary = ", ".join(f"{k} {v}" for k, v in sorted(got.items())) or "nothing new"
            print(f"{breed}: {summary}")
    except RateLimitedError as exc:
        print(exc)
    print("Next: make data-build, make embed, make screen, make train.")


if __name__ == "__main__":
    main()
