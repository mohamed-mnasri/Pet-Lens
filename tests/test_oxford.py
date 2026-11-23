"""End-to-end Oxford build on a tiny fake copy of the dataset (no network)."""

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from petlens.data import oxford
from petlens.data.catalog import Source, SourceFile, load_breeds
from petlens.data.download import ChecksumError, fetch
from petlens.data.manifest import read_manifest

# name, species code, split
ENTRIES = [
    ("Abyssinian_1", "1", "trainval"),
    ("Abyssinian_2", "1", "test"),
    ("beagle_1", "2", "trainval"),
    ("beagle_2", "2", "test"),  # exact copy of beagle_1
    ("Bombay_1", "1", "trainval"),  # corrupt file
]


def _jpeg(seed: int, size: tuple[int, int] = (320, 240)) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def _trimap(size: tuple[int, int] = (320, 240)) -> bytes:
    arr = np.full((size[1], size[0]), 2, dtype=np.uint8)
    arr[60:180, 80:240] = 1
    arr[59, 80:240] = 3
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _make_archives(root: Path) -> list[SourceFile]:
    images = root / "images.tar.gz"
    with tarfile.open(images, "w:gz") as tar:
        _add(tar, "images/Abyssinian_1.jpg", _jpeg(1))
        _add(tar, "images/Abyssinian_2.jpg", _jpeg(2))
        _add(tar, "images/beagle_1.jpg", _jpeg(3))
        _add(tar, "images/beagle_2.jpg", _jpeg(3))
        _add(tar, "images/Bombay_1.jpg", b"\xff\xd8 truncated")
    annotations = root / "annotations.tar.gz"
    with tarfile.open(annotations, "w:gz") as tar:
        for split in ("trainval", "test"):
            lines = ["#Image CLASS-ID SPECIES BREED ID"]
            lines += [f"{n} 1 {sp} 1" for n, sp, s in ENTRIES if s == split]
            _add(tar, f"annotations/{split}.txt", "\n".join(lines).encode())
        for n, _, _ in ENTRIES:
            if n != "Abyssinian_2":  # one image without a trimap
                _add(tar, f"annotations/trimaps/{n}.png", _trimap())
    specs = []
    for p in (images, annotations):
        md5 = hashlib.md5(p.read_bytes(), usedforsecurity=False).hexdigest()
        specs.append(SourceFile(name=p.name, md5=md5, urls=[p.as_uri()]))
    return specs


@pytest.fixture
def source(tmp_path: Path) -> Source:
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    return Source(
        name="Oxford-IIIT Pet",
        version="test",
        homepage="https://example.org",
        licence="CC BY-SA 4.0",
        attribution="Oxford-IIIT Pet test fixture",
        files=_make_archives(mirror),
    )


def test_build_manifest(tmp_path: Path, source: Source) -> None:
    data_dir = tmp_path / "data"
    manifest = oxford.build(data_dir, load_breeds(), source)
    rows = {r.source_ref: r for r in read_manifest(manifest)}

    assert set(rows) == {
        "images/Abyssinian_1.jpg",
        "images/Abyssinian_2.jpg",
        "images/beagle_1.jpg",
    }
    a1 = rows["images/Abyssinian_1.jpg"]
    assert (a1.species, a1.breed, a1.label_origin, a1.source_split) == (
        "cat",
        "abyssinian",
        "oxford",
        "trainval",
    )
    assert a1.licence == "CC BY-SA 4.0" and a1.attribution
    assert a1.bbox == pytest.approx([0.25, 59 / 240, 0.75, 0.75], abs=1e-3)
    assert a1.split is None  # splits are assigned in a later stage
    assert rows["images/Abyssinian_2.jpg"].flags == ["no_trimap"]
    assert rows["images/beagle_1.jpg"].species == "dog"

    report = (manifest.parent / "report.md").read_text()
    assert "beagle_2 = beagle_1" in report
    assert "Bombay_1: cannot decode" in report
    for r in rows.values():
        assert (data_dir / "processed" / "images" / r.image_id[:2] / f"{r.image_id}.webp").exists()


def test_rerun_is_identical(tmp_path: Path, source: Source) -> None:
    data_dir = tmp_path / "data"
    first = oxford.build(data_dir, load_breeds(), source).read_bytes()
    second = oxford.build(data_dir, load_breeds(), source).read_bytes()
    assert first == second


def test_parallel_matches_serial(tmp_path: Path, source: Source) -> None:
    serial = oxford.build(tmp_path / "a", load_breeds(), source)
    parallel = oxford.build(tmp_path / "b", load_breeds(), source, workers=2)
    assert read_manifest(serial) == read_manifest(parallel)


def test_unknown_oxford_label_fails_loudly(tmp_path: Path, source: Source) -> None:
    ann = tmp_path / "x" / "annotations"
    ann.mkdir(parents=True)
    (ann / "trainval.txt").write_text("Tabby_1 1 1 1\n")
    (ann / "test.txt").write_text("")
    with pytest.raises(ValueError, match=r"not in breeds\.yaml"):
        oxford.ingest(tmp_path / "x", tmp_path / "img", load_breeds(), source)


def test_fetch_rejects_bad_checksum(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    spec = SourceFile(name="a.bin", md5="0" * 32, urls=[f.as_uri()])
    with pytest.raises(ChecksumError):
        fetch(spec, tmp_path / "out")
    assert not (tmp_path / "out" / "a.bin").exists()
