"""Commons pipeline against a fake API (no network)."""

import io
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from petlens.data import commons
from petlens.data.catalog import BreedCatalog, Source
from petlens.data.commons import CommonsClient, normalise_licence, strip_html
from petlens.data.manifest import read_manifest

BREEDS = BreedCatalog.model_validate(
    {
        "breeds": [
            {
                "id": "ragdoll",
                "name": "Ragdoll",
                "species": "cat",
                "role": "train",
                "commons_categories": ["Ragdoll cats"],
            },
            {
                "id": "birman",
                "name": "Birman",
                "species": "cat",
                "role": "train",
                "commons_categories": ["Birman cats"],
            },
            {"id": "beagle", "name": "Beagle", "species": "dog", "role": "train"},
        ]
    }
)
SOURCE = Source(
    name="Wikimedia Commons",
    version="live",
    homepage="https://commons.wikimedia.org/",
    licence="per file",
    attribution="per file",
    licence_allowlist=["CC0", "Public domain", "CC BY", "CC BY-SA"],
)


def _jpeg(seed: int) -> bytes:
    arr = np.random.default_rng(seed).integers(40, 256, size=(300, 400, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _file(
    title: str,
    user: str,
    licence: str = "CC BY-SA 4.0",
    seed: int = 0,
    width: int = 4000,
    height: int = 3000,
    mime: str = "image/jpeg",
    artist: str = '<a href="//commons.wikimedia.org/wiki/User:X">Jane &amp; Co</a>',
) -> dict[str, Any]:
    return {
        "ns": 6,
        "title": title,
        "imageinfo": [
            {
                "user": user,
                "width": width,
                "height": height,
                "mime": mime,
                "thumburl": f"https://upload.example/{seed}.jpg",
                "descriptionurl": f"https://commons.wikimedia.org/wiki/{title.replace(' ', '_')}",
                "extmetadata": {
                    "LicenseShortName": {"value": licence},
                    "Artist": {"value": artist},
                },
            }
        ],
    }


def _sub(name: str) -> dict[str, Any]:
    return {"ns": 14, "title": f"Category:{name}"}


# Each category maps to a list of pages (each page is one API response).
TREE: dict[str, list[list[dict[str, Any]]]] = {
    "Category:Ragdoll cats": [
        [
            _file("File:R1.jpg", "alice", seed=1),
            _file("File:R2.jpg", "alice", seed=2),
            _sub("Ragdoll kittens"),
        ],
        [
            _file("File:R3.jpg", "bob", "CC BY-NC-SA 2.0", seed=3),
            _sub("Drawings of Ragdoll cats"),
            _file("File:Both.jpg", "carol", seed=9),
        ],
    ],
    "Category:Ragdoll kittens": [
        [_file("File:RK1.jpg", "dave", "Public domain", seed=4), _sub("Ragdoll kittens deep")],
    ],
    "Category:Ragdoll kittens deep": [[_file("File:Deep.jpg", "erin", seed=5)]],
    "Category:Drawings of Ragdoll cats": [[_file("File:Drawing.png", "fay", seed=6)]],
    "Category:Birman cats": [
        [
            _file("File:B1.jpg", "gus", "CC0", seed=7),
            _file("File:Tiny.jpg", "gus", seed=8, width=200, height=150),
            _file("File:B9-copy.jpg", "hal", seed=7),  # same bytes as B1
            _file("File:Both.jpg", "carol", seed=9),
        ],
    ],
}


class FakeApi:
    def __init__(self) -> None:
        self.json_calls = 0
        self.byte_calls = 0

    def get_json(self, params: dict[str, str]) -> dict[str, Any]:
        self.json_calls += 1
        pages = TREE[params["gcmtitle"]]
        idx = int(params.get("gcmcontinue", "0"))
        resp: dict[str, Any] = {"query": {"pages": pages[idx]}}
        if idx + 1 < len(pages):
            resp["continue"] = {"gcmcontinue": str(idx + 1), "continue": "gcmcontinue||"}
        return resp

    def get_bytes(self, url: str) -> bytes:
        self.byte_calls += 1
        return _jpeg(int(url.rsplit("/", 1)[1].split(".")[0]))


def _client(tmp_path: Path, api: FakeApi) -> CommonsClient:
    return CommonsClient(
        tmp_path / "cache",
        min_interval=0,
        get_json=api.get_json,
        get_bytes=api.get_bytes,
        sleep=lambda _: None,
    )


@pytest.mark.parametrize(
    ("raw", "family"),
    [
        ("CC BY-SA 4.0", "CC BY-SA"),
        ("CC BY 2.0", "CC BY"),
        ("CC0", "CC0"),
        ("Public domain", "Public domain"),
        ("PD-self", "Public domain"),
        ("CC BY-NC-SA 2.0", None),
        ("CC BY-ND 4.0", None),
        ("GFDL", None),
        ("", None),
    ],
)
def test_normalise_licence(raw: str, family: str | None) -> None:
    assert normalise_licence(raw) == family


def test_strip_html() -> None:
    assert strip_html('<a href="x">Jane &amp; <b>Co</b></a>\n') == "Jane & Co"


def test_build(tmp_path: Path) -> None:
    api = FakeApi()
    manifest = commons.build(tmp_path / "data", BREEDS, SOURCE, client=_client(tmp_path, api))
    rows = {r.source_ref.rsplit("/", 1)[1]: r for r in read_manifest(manifest)}

    # R3: non-commercial licence; Drawing: art subcategory; Deep: beyond depth 1;
    # Tiny: too small; B9-copy: exact duplicate of B1 (first title wins);
    # Both: filed under two labels.
    assert set(rows) == {"File:R1.jpg", "File:R2.jpg", "File:RK1.jpg", "File:B1.jpg"}
    r1 = rows["File:R1.jpg"]
    assert (r1.species, r1.breed, r1.label_origin, r1.uploader) == (
        "cat",
        "ragdoll",
        "commons_category",
        "alice",
    )
    assert r1.licence == "CC BY-SA 4.0"
    assert r1.attribution.startswith("Jane & Co, CC BY-SA 4.0, via Wikimedia Commons (https://")
    assert r1.split is None and r1.bbox is None

    report = (manifest.parent / "report.md").read_text()
    assert "licence_not_allowed: 1" in report
    assert "exact_duplicate: 1" in report
    assert "too_small: 1" in report
    assert "conflicting_labels: 1" in report
    assert "File:Both.jpg: birman, ragdoll" in report
    assert "Drawings of Ragdoll cats" in report


def test_rerun_uses_cache(tmp_path: Path) -> None:
    api = FakeApi()
    first = commons.build(tmp_path / "data", BREEDS, SOURCE, client=_client(tmp_path, api))
    calls = (api.json_calls, api.byte_calls)
    first_bytes = first.read_bytes()
    second = commons.build(tmp_path / "data", BREEDS, SOURCE, client=_client(tmp_path, api))
    assert (api.json_calls, api.byte_calls) == calls  # nothing fetched again
    assert second.read_bytes() == first_bytes


def test_cap_and_depth(tmp_path: Path) -> None:
    api = FakeApi()
    manifest = commons.build(
        tmp_path / "data",
        BREEDS,
        SOURCE,
        max_per_category=1,
        max_depth=2,
        client=_client(tmp_path, api),
    )
    breeds = [r.breed for r in read_manifest(manifest)]
    assert sorted(breeds) == ["birman", "ragdoll"]


def test_deeper_walk_finds_more(tmp_path: Path) -> None:
    files, skipped = commons.walk_category(_client(tmp_path, FakeApi()), "Ragdoll cats", 2)
    assert "File:Deep.jpg" in {f.title for f in files}
    assert skipped == ["Drawings of Ragdoll cats"]


def test_retries_on_429(tmp_path: Path) -> None:
    attempts = {"n": 0}
    sleeps: list[float] = []

    def flaky(params: dict[str, str]) -> dict[str, Any]:
        attempts["n"] += 1
        if attempts["n"] < 3:
            headers = Message()
            headers["Retry-After"] = "7"
            raise urllib.error.HTTPError("u", 429, "Too Many Requests", headers, None)
        return {"query": {"pages": []}}

    client = CommonsClient(tmp_path, min_interval=0, get_json=flaky, sleep=sleeps.append)
    assert client.query({"list": "x"}) == {"query": {"pages": []}}
    assert attempts["n"] == 3
    assert sleeps.count(7.0) == 2


def test_api_error_raises(tmp_path: Path) -> None:
    client = CommonsClient(
        tmp_path,
        min_interval=0,
        get_json=lambda p: {"error": {"code": "badvalue"}},
        sleep=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="badvalue"):
        client.query({"list": "x"})
