"""Wikimedia Commons: walk cat breed categories, keep reusable files, write the manifest.

Category names are weak labels. Every kept file records its author, licence, page URL
and uploader (used for grouped splits in part 2c). Only the 1024 px thumbnail is
downloaded, because that is all PetLens stores anyway.

API etiquette (https://www.mediawiki.org/wiki/API:Etiquette): a descriptive User-Agent,
one request at a time with a minimum interval, ``maxlag``, backoff on 429/5xx, and an
on-disk cache so a rerun doesn't fetch anything twice.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from petlens.data.catalog import Breed, BreedCatalog, Source
from petlens.data.download import USER_AGENT
from petlens.data.imaging import DEFAULT_POLICY, ImageRejectedError, StoragePolicy, store_image
from petlens.data.manifest import ManifestRow, write_manifest

log = logging.getLogger(__name__)

SOURCE_KEY = "commons"
API_URL = "https://commons.wikimedia.org/w/api.php"
THUMB_WIDTH = 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/tiff"}

# Subcategories whose names match this are skipped: they hold art, not photos of cats.
DEFAULT_EXCLUDE = re.compile(
    r"\b(drawings?|paintings?|art|artworks?|illustrations?|cartoons?|sculptures?|statues?|"
    r"figurines?|toys?|plush|logos?|stamps?|postcards?|coats? of arms|icons?|diagrams?|"
    r"skeletons?|taxidermy|videos?|audio)\b",
    re.IGNORECASE,
)

JsonGetter = Callable[[dict[str, str]], dict[str, Any]]
BytesGetter = Callable[[str], bytes]


# --------------------------------------------------------------------------- HTTP client


class CommonsClient:
    """Throttled, cached access to the MediaWiki API and thumbnail files."""

    def __init__(
        self,
        cache_dir: Path,
        min_interval: float = 1.0,
        max_retries: int = 5,
        get_json: JsonGetter | None = None,
        get_bytes: BytesGetter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cache_dir = cache_dir
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._get_json = get_json or self._http_json
        self._get_bytes = get_bytes or self._http_bytes
        self._sleep = sleep
        self._last = 0.0
        self.requests_made = 0

    # -- transport (replaced in tests)

    def _open(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data: bytes = resp.read()
            return data

    def _http_json(self, params: dict[str, str]) -> dict[str, Any]:
        url = API_URL + "?" + urllib.parse.urlencode(sorted(params.items()))
        result: dict[str, Any] = json.loads(self._open(url))
        return result

    def _http_bytes(self, url: str) -> bytes:
        return self._open(url)

    # -- throttle and retry

    def _throttled(self, fn: Callable[[], Any]) -> Any:
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = time.monotonic()
            self.requests_made += 1
            try:
                return fn()
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = (
                    float(retry_after) if retry_after and retry_after.isdigit() else 2.0**attempt
                )
                log.warning("HTTP %s, retrying in %.0f s", exc.code, delay)
                self._sleep(delay)
        raise AssertionError("unreachable")

    # -- public API

    def query(self, params: dict[str, str]) -> dict[str, Any]:
        full = {"action": "query", "format": "json", "formatversion": "2", "maxlag": "5", **params}
        key = hashlib.sha256(json.dumps(full, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / "api" / key[:2] / f"{key}.json"
        if path.exists():
            cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return cached
        for _ in range(self.max_retries + 1):
            data: dict[str, Any] = self._throttled(lambda: self._get_json(full))
            if data.get("error", {}).get("code") == "maxlag":
                self._sleep(5.0)
                continue
            break
        if "error" in data:
            raise RuntimeError(f"Commons API error: {data['error']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def query_all(self, params: dict[str, str]) -> Iterator[dict[str, Any]]:
        """Follow ``continue`` tokens, yielding each response page."""
        cont: dict[str, str] = {}
        while True:
            page = self.query({**params, **cont})
            yield page
            if "continue" not in page:
                return
            cont = {k: str(v) for k, v in page["continue"].items()}

    def download(self, url: str) -> bytes:
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache_dir / "files" / key[:2] / key
        if path.exists():
            return path.read_bytes()
        data: bytes = self._throttled(lambda: self._get_bytes(url))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return data


# --------------------------------------------------------------------------- metadata


def normalise_licence(short_name: str) -> str | None:
    """Map Commons ``LicenseShortName`` to an allowlist family, or None if not reusable."""
    s = short_name.strip().lower().replace("-", " ")
    if re.search(r"\bnc\b|\bnd\b|non commercial|no derivatives", s):
        return None
    if s.startswith("cc0") or s.startswith("cc zero"):
        return "CC0"
    if s.startswith("public domain") or s.startswith("pd") or s == "no restrictions":
        return "Public domain"
    if s.startswith("cc by sa"):
        return "CC BY-SA"
    if s.startswith("cc by"):
        return "CC BY"
    return None


def strip_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


@dataclass(frozen=True)
class CommonsFile:
    title: str
    category: str
    page_url: str
    thumb_url: str
    mime: str
    width: int
    height: int
    uploader: str
    author: str
    licence: str  # as written on Commons, e.g. "CC BY-SA 4.0"
    licence_family: str | None


def parse_page(page: dict[str, Any], category: str) -> CommonsFile | None:
    infos = page.get("imageinfo") or []
    if not infos:
        return None
    info = infos[0]
    meta = info.get("extmetadata", {})

    def m(key: str) -> str:
        return str(meta.get(key, {}).get("value", ""))

    licence = strip_html(m("LicenseShortName"))
    author = strip_html(m("Artist")) or str(info.get("user", ""))
    return CommonsFile(
        title=str(page["title"]),
        category=category,
        page_url=str(info.get("descriptionurl", "")),
        thumb_url=str(info.get("thumburl") or info.get("url", "")),
        mime=str(info.get("mime", "")),
        width=int(info.get("width", 0)),
        height=int(info.get("height", 0)),
        uploader=str(info.get("user", "")),
        author=author,
        licence=licence,
        licence_family=normalise_licence(licence),
    )


# --------------------------------------------------------------------------- discovery


def walk_category(
    client: CommonsClient,
    root: str,
    max_depth: int = 1,
    exclude: re.Pattern[str] = DEFAULT_EXCLUDE,
) -> tuple[list[CommonsFile], list[str]]:
    """Files in ``root`` and its subcategories up to ``max_depth``, plus skipped subcats."""
    files: dict[str, CommonsFile] = {}
    skipped: list[str] = []
    seen_cats: set[str] = set()
    queue: list[tuple[str, int]] = [(f"Category:{root}", 0)]
    while queue:
        cat, depth = queue.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        for page in client.query_all(
            {
                "generator": "categorymembers",
                "gcmtitle": cat,
                "gcmtype": "file|subcat",
                "gcmlimit": "50",
                "prop": "imageinfo",
                "iiprop": "url|size|mime|user|extmetadata",
                "iiurlwidth": str(THUMB_WIDTH),
                "iiextmetadatafilter": "LicenseShortName|Artist",
            }
        ):
            for p in page.get("query", {}).get("pages", []):
                title = str(p.get("title", ""))
                if p.get("ns") == 14:
                    name = title.removeprefix("Category:")
                    if exclude.search(name):
                        skipped.append(name)
                    elif depth < max_depth:
                        queue.append((title, depth + 1))
                elif p.get("ns") == 6 and title not in files:
                    f = parse_page(p, root)
                    if f is not None:
                        files[title] = f
    return [files[k] for k in sorted(files)], sorted(set(skipped))


# --------------------------------------------------------------------------- build


@dataclass
class CommonsReport:
    candidates: dict[str, int] = field(default_factory=dict)
    kept: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    licences: dict[str, int] = field(default_factory=dict)
    uploaders: dict[str, int] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    excluded_subcategories: dict[str, list[str]] = field(default_factory=dict)
    empty_categories: list[str] = field(default_factory=list)
    stored_bytes: int = 0
    requests_made: int = 0


def build_rows(
    client: CommonsClient,
    breeds: BreedCatalog,
    source: Source,
    images_dir: Path,
    max_per_category: int | None = None,
    max_depth: int = 1,
    policy: StoragePolicy = DEFAULT_POLICY,
) -> tuple[list[ManifestRow], CommonsReport]:
    report = CommonsReport()
    allow = set(source.licence_allowlist)
    skip: Counter[str] = Counter()

    # 1. discover, and find files filed under more than one label
    by_title: dict[str, list[tuple[Breed, CommonsFile]]] = {}
    for breed in breeds.breeds:
        for category in breed.commons_categories:
            files, excluded = walk_category(client, category, max_depth)
            report.excluded_subcategories[category] = excluded
            report.candidates[category] = len(files)
            if not files:
                log.warning("Commons category %r has no usable files; check breeds.yaml", category)
                report.empty_categories.append(category)
            for f in files:
                by_title.setdefault(f.title, []).append((breed, f))

    chosen: dict[str, list[tuple[Breed, CommonsFile]]] = {}
    for title in sorted(by_title):
        labels = {b.id for b, _ in by_title[title]}
        if len(labels) > 1:
            report.conflicts.append(f"{title}: {', '.join(sorted(labels))}")
            skip["conflicting_labels"] += 1
            continue
        breed, f = by_title[title][0]
        chosen.setdefault(breed.id, []).append((breed, f))

    # 2. filter and store
    rows: list[ManifestRow] = []
    seen_ids: set[str] = set()
    flags: Counter[str] = Counter()
    licences: Counter[str] = Counter()
    for breed_id in sorted(chosen):
        kept = 0
        for breed, f in chosen[breed_id]:
            if max_per_category is not None and kept >= max_per_category:
                skip["over_per_category_cap"] += 1
                continue
            reason = _skip_reason(f, allow, policy)
            if reason:
                skip[reason] += 1
                continue
            try:
                stored = store_image(client.download(f.thumb_url), images_dir, policy)
            except ImageRejectedError:
                skip["cannot_decode"] += 1
                continue
            except OSError as exc:
                log.warning("download failed for %s: %s", f.title, exc)
                skip["download_failed"] += 1
                continue
            if stored.image_id in seen_ids:
                skip["exact_duplicate"] += 1
                continue
            seen_ids.add(stored.image_id)
            kept += 1
            rows.append(
                ManifestRow(
                    image_id=stored.image_id,
                    source=SOURCE_KEY,
                    source_ref=f.page_url or f.title,
                    licence=f.licence,
                    attribution=f"{f.author}, {f.licence}, via Wikimedia Commons ({f.page_url})",
                    species=breed.species,
                    breed=breed.id,
                    label_origin="commons_category",
                    width=stored.width,
                    height=stored.height,
                    uploader=f.uploader,
                    author=f.author,
                    flags=list(stored.flags),
                )
            )
            flags.update(stored.flags)
            licences[f.licence_family or "?"] += 1
            report.stored_bytes += stored.nbytes
        report.kept[breed_id] = kept

    report.uploaders = {b: len({r.uploader for r in rows if r.breed == b}) for b in sorted(chosen)}
    report.skipped = dict(sorted(skip.items()))
    report.flags = dict(sorted(flags.items()))
    report.licences = dict(sorted(licences.items()))
    report.requests_made = client.requests_made
    return rows, report


def _skip_reason(f: CommonsFile, allow: set[str], policy: StoragePolicy) -> str | None:
    if f.licence_family is None or f.licence_family not in allow:
        return "licence_not_allowed"
    if not f.author:
        return "no_author"
    if f.mime not in ALLOWED_MIME:
        return "not_a_photo_format"
    if min(f.width, f.height) < policy.min_short_side:
        return "too_small"
    if not f.thumb_url:
        return "no_url"
    return None


def render_report(report: CommonsReport) -> str:
    total_kept = sum(report.kept.values())
    lines = [
        "# Wikimedia Commons ingest report",
        "",
        f"- Files kept: {total_kept}",
        f"- API and file requests made this run: {report.requests_made} (cached ones not counted)",
        f"- Stored size: {report.stored_bytes / 1e6:.1f} MB",
        "",
        "## Per label",
        "",
        "| Label | Kept | Uploaders |",
        "| --- | ---: | ---: |",
    ]
    for breed, n in report.kept.items():
        lines.append(f"| {breed} | {n} | {report.uploaders.get(breed, 0)} |")
    lines += ["", "## Candidates per category", "", "| Category | Files found |", "| --- | ---: |"]
    lines += [f"| {c} | {n} |" for c, n in report.candidates.items()]
    lines += ["", "## Skipped", ""]
    lines += [f"- {k}: {v}" for k, v in report.skipped.items()] or ["- none"]
    lines += ["", "## Licences of kept files", ""]
    lines += [f"- {k}: {v}" for k, v in report.licences.items()] or ["- none"]
    lines += ["", "## Flags", ""]
    lines += [f"- {k}: {v}" for k, v in report.flags.items()] or ["- none"]
    if report.empty_categories:
        lines += ["", "## Empty categories (check the names in breeds.yaml)", ""]
        lines += [f"- {c}" for c in report.empty_categories]
    excluded = {c: s for c, s in report.excluded_subcategories.items() if s}
    if excluded:
        lines += ["", "## Subcategories skipped as art, toys and similar", ""]
        for c, subs in excluded.items():
            lines.append(f"- {c}: {', '.join(subs)}")
    if report.conflicts:
        lines += ["", "## Files under more than one label (skipped)", ""]
        lines += [f"- {c}" for c in report.conflicts]
    return "\n".join(lines) + "\n"


def build(
    data_dir: Path,
    breeds: BreedCatalog,
    source: Source,
    max_per_category: int | None = None,
    max_depth: int = 1,
    client: CommonsClient | None = None,
) -> Path:
    """Discover, filter, download and store. Returns the manifest path. Safe to rerun."""
    client = client or CommonsClient(data_dir / "raw" / SOURCE_KEY / "cache")
    rows, report = build_rows(
        client, breeds, source, data_dir / "processed" / "images", max_per_category, max_depth
    )
    out_dir = data_dir / "processed" / "sources" / SOURCE_KEY
    manifest = out_dir / "manifest.parquet"
    write_manifest(rows, manifest)
    (out_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report.__dict__, indent=2), encoding="utf-8")
    log.info("commons: kept %d files", len(rows))
    return manifest
