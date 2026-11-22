"""Dataset version build: near-duplicates, groups, splits and stability."""

import io
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from petlens.data import dataset
from petlens.data.catalog import load_breeds
from petlens.data.dataset import partition
from petlens.data.imaging import store_image
from petlens.data.manifest import ManifestRow, read_manifest, write_manifest
from petlens.data.phash import near_duplicate_pairs, phash_image

BREEDS = load_breeds()


def _photo(seed: int, brighten: float = 1.0) -> bytes:
    # Smooth random blobs, so a brightness change is a near-duplicate, not a new image.
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize((400, 300), Image.Resampling.BICUBIC)
    if brighten != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brighten)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class Fixture:
    def __init__(self, root: Path) -> None:
        self.data = root / "data"
        self.repo = root / "repo"
        self.images = self.data / "processed" / "images"
        self.rows: dict[str, list[ManifestRow]] = defaultdict(list)

    def add(
        self,
        source: str,
        ref: str,
        breed: str,
        seed: int,
        *,
        brighten: float = 1.0,
        source_split: str | None = None,
        uploader: str | None = None,
        author: str | None = None,
        species: str = "cat",
    ) -> str:
        stored = store_image(_photo(seed, brighten), self.images)
        self.rows[source].append(
            ManifestRow(
                image_id=stored.image_id,
                source=source,
                source_ref=ref,
                licence="CC BY 4.0",
                attribution=f"author of {ref}",
                species=species,
                breed=breed,
                label_origin=source,
                width=stored.width,
                height=stored.height,
                source_split=source_split,
                uploader=uploader,
                author=author,
            )
        )
        return stored.image_id

    def write(self) -> None:
        for source, rows in self.rows.items():
            write_manifest(rows, self.data / "processed" / "sources" / source / "manifest.parquet")

    def build(self, day: date = date(2026, 11, 22)) -> dataset.BuildResult:
        self.write()
        return dataset.build(self.data, BREEDS, repo_data_dir=self.repo, today=day)


def _populate(fx: Fixture) -> None:
    seed = 0
    for breed in ("abyssinian", "bengal", "ragdoll"):
        for k in range(12):
            seed += 1
            fx.add(
                "oxford",
                f"images/{breed}_{k}.jpg",
                breed,
                seed,
                source_split="test" if k % 3 == 0 else "trainval",
            )
    for u in range(30):
        for k in range(3):
            seed += 1
            breed = "domestic_cat" if u % 2 else "persian"
            fx.add("commons", f"File:u{u}_{k}.jpg", breed, seed, uploader=f"user{u}")
    for k in range(4):
        seed += 1
        fx.add("commons", f"File:fold_{k}.jpg", "scottish_fold", seed, uploader="user0")


def _by_ref(path: Path) -> dict[str, ManifestRow]:
    return {r.source_ref: r for r in read_manifest(path)}


def test_phash_near_duplicates() -> None:
    a = phash_image(Image.open(io.BytesIO(_photo(1))))
    b = phash_image(Image.open(io.BytesIO(_photo(1, brighten=1.15))))
    c = phash_image(Image.open(io.BytesIO(_photo(2))))
    hashes = [f"{a:016x}", f"{b:016x}", f"{c:016x}"]
    assert near_duplicate_pairs(hashes, 6) == [(0, 1)]


def test_no_group_crosses_splits_and_every_row_is_attributed(tmp_path: Path) -> None:
    fx = Fixture(tmp_path)
    _populate(fx)
    result = fx.build()
    rows = read_manifest(result.manifest)
    assert result.changed and result.version.startswith("2026-11-22-")

    parts: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        assert r.licence and r.attribution
        assert r.split in dataset.SPLITS
        assert r.added_in == result.version
        if r.split not in ("excluded", "test_unknown"):
            parts[r.group_id or ""].add(partition(r.split or ""))
    assert all(len(p) == 1 for p in parts.values())

    splits = {r.split for r in rows}
    assert {"train", "test_in_domain", "test_unknown"} <= splits
    assert splits & {"test_unseen_source", "test_domestic", "rw_calibration"}
    for r in rows:
        if r.source == "oxford":
            expected = (
                {"test_in_domain"} if r.source_split == "test" else {"train", "val", "calibration"}
            )
            assert r.split in expected
        if r.breed == "scottish_fold":
            assert r.split == "test_unknown"
        if r.split == "test_domestic":
            assert r.breed == "domestic_cat"
    # One uploader's photos share a group.
    assert len({r.group_id for r in rows if r.uploader == "user3"}) == 1


def test_leaks_are_excluded(tmp_path: Path) -> None:
    fx = Fixture(tmp_path)
    _populate(fx)
    fx.add("oxford", "images/bengal_train_leak.jpg", "bengal", 9001, source_split="trainval")
    fx.add(
        "oxford", "images/bengal_test_leak.jpg", "bengal", 9001, brighten=1.1, source_split="test"
    )
    fx.add("oxford", "images/ragdoll_test.jpg", "ragdoll", 9002, source_split="test")
    fx.add("commons", "File:copy_of_test.jpg", "ragdoll", 9002, brighten=0.9, uploader="zed")
    rows = _by_ref(fx.build().manifest)
    assert rows["images/bengal_test_leak.jpg"].split == "excluded"
    assert "leak_train_test" in rows["images/bengal_test_leak.jpg"].flags
    assert rows["images/bengal_train_leak.jpg"].split in {"train", "val", "calibration"}
    assert rows["images/ragdoll_test.jpg"].split == "test_in_domain"
    assert rows["File:copy_of_test.jpg"].split == "excluded"
    assert "dup_of_test" in rows["File:copy_of_test.jpg"].flags


def test_rebuild_without_changes_keeps_version(tmp_path: Path) -> None:
    fx = Fixture(tmp_path)
    _populate(fx)
    first = fx.build()
    before = first.manifest.read_bytes()
    second = fx.build(day=date(2026, 12, 1))
    assert not second.changed
    assert second.version == first.version
    assert first.manifest.read_bytes() == before
    assert (fx.repo / "manifests" / "LATEST").read_text().strip() == first.version


def test_existing_assignments_never_move(tmp_path: Path) -> None:
    fx = Fixture(tmp_path)
    _populate(fx)
    first = fx.build()
    old = _by_ref(first.manifest)

    # New images, some from uploaders already in the dataset, one from a new uploader.
    fx.add("commons", "File:u3_new.jpg", "domestic_cat", 7001, uploader="user3")
    fx.add("commons", "File:u4_new.jpg", "persian", 7002, uploader="user4")
    fx.add("commons", "File:newbie.jpg", "persian", 7003, uploader="newbie")
    second = fx.build(day=date(2026, 12, 1))
    assert second.changed and second.version.startswith("2026-12-01-")
    new = _by_ref(second.manifest)

    for ref, r in old.items():
        assert new[ref].split == r.split
        assert new[ref].added_in == first.version
    assert new["File:newbie.jpg"].added_in == second.version
    for ref, uploader in (("File:u3_new.jpg", "user3"), ("File:u4_new.jpg", "user4")):
        group_split = {partition(r.split or "") for r in old.values() if r.uploader == uploader}
        assert {partition(new[ref].split or "")} == group_split


def test_needs_sources(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="make data-oxford"):
        dataset.build(tmp_path, BREEDS, repo_data_dir=tmp_path / "repo")


def test_bulk_uploader_is_split_by_author(tmp_path: Path) -> None:
    fx = Fixture(tmp_path)
    _populate(fx)
    # One transfer account uploads two photographers' work; generic authors fall back.
    for k in range(3):
        fx.add(
            "commons", f"File:alice_{k}.jpg", "persian", 8000 + k, uploader="bot", author="Alice"
        )
        fx.add("commons", f"File:bob_{k}.jpg", "bengal", 8100 + k, uploader="bot", author="Bob B.")
        fx.add(
            "commons",
            f"File:anon_{k}.jpg",
            "persian",
            8200 + k,
            uploader="bot",
            author="Unknown author",
        )
    rows = _by_ref(fx.build().manifest)
    groups = {
        who: {rows[f"File:{who}_{k}.jpg"].group_id for k in range(3)}
        for who in ("alice", "bob", "anon")
    }
    assert all(len(g) == 1 for g in groups.values())
    assert len(set().union(*groups.values())) == 3
