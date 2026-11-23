"""Cached, checksum-verified downloads and safe archive extraction."""

from __future__ import annotations

import hashlib
import logging
import tarfile
import urllib.request
from pathlib import Path

from petlens import __version__
from petlens.data.catalog import SourceFile

log = logging.getLogger(__name__)

USER_AGENT = (
    f"PetLens/{__version__} (local research project; +https://github.com/mohamed-mnasri/petlens)"
)
_CHUNK = 1 << 20


class ChecksumError(RuntimeError):
    pass


def md5_of(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(spec: SourceFile, dest_dir: Path, timeout: float = 60.0) -> Path:
    """Download ``spec`` into ``dest_dir`` unless a verified copy is already there.

    Mirrors are tried in order. A file only gets its final name after its MD5 matches,
    so an interrupted download never looks complete.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / spec.name
    if target.exists():
        if md5_of(target) == spec.md5:
            log.info("cached: %s", target)
            return target
        log.warning("checksum mismatch on cached %s, downloading again", target)
        target.unlink()

    part = target.with_suffix(target.suffix + ".part")
    errors: list[str] = []
    for url in spec.urls:
        log.info("downloading %s", url)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as out:
                for chunk in iter(lambda: resp.read(_CHUNK), b""):
                    out.write(chunk)
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            part.unlink(missing_ok=True)
            continue
        got = md5_of(part)
        if got != spec.md5:
            errors.append(f"{url}: md5 {got}, expected {spec.md5}")
            part.unlink()
            continue
        part.replace(target)
        return target
    raise ChecksumError(f"could not fetch {spec.name}: " + "; ".join(errors))


def extract(archive: Path, dest_dir: Path) -> Path:
    """Extract a tar archive once. A marker file makes reruns a no-op."""
    marker = dest_dir / f".extracted-{archive.name}"
    if marker.exists():
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        # The "data" filter rejects absolute paths, links outside dest_dir and device files.
        tar.extractall(dest_dir, filter="data")
    marker.touch()
    return dest_dir
