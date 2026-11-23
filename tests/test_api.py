"""Upload checks, storage limits and the HTTP API (with a fake analyzer, no real models).

Signature, size and pixel checks run without a database. The API tests need Postgres
(CI has it) and are skipped unless DATABASE_URL is set; they recreate the schema.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, text

from petlens.api import ingest
from petlens.api.app import create_app
from petlens.config import Settings
from petlens.contracts import EMBEDDING_DIM
from petlens.db import migrate
from petlens.models.heads import train_species_head
from petlens.models.openset import KnnIndex
from petlens.models.policy import PolicyConfig, SpeciesThresholds
from petlens.vision.detect import Detection
from petlens.vision.pipeline import Analyzer

DATABASE_URL = os.environ.get("DATABASE_URL")
RED = (220, 40, 40)


def _jpeg(size: tuple[int, int], colour: tuple[int, int, int] = (128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _png(size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 10, 10)).save(buf, "PNG")
    return buf.getvalue()


def _clip(path: Path, seconds: int, fps: int = 2) -> bytes:
    import av

    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        for _ in range(seconds * fps):
            img = Image.new("RGB", (160, 120), (128, 128, 128))
            img.paste(RED, (50, 30, 110, 90))
            for packet in stream.encode(av.VideoFrame.from_image(img)):  # type: ignore[no-untyped-call]
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path.read_bytes()


# --- checks that need no database -------------------------------------------------


@pytest.mark.parametrize(
    ("data", "kind", "mime"),
    [
        (_jpeg((8, 8)), "photo", "image/jpeg"),
        (_png((8, 8)), "photo", "image/png"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "photo", "image/webp"),
        (b"\x00\x00\x00\x18ftypisom\x00\x00", "video", "video/mp4"),
        (b"\x00\x00\x00\x14ftypqt  \x00\x00", "video", "video/quicktime"),
        (b"\x1a\x45\xdf\xa3\x01\x00", "video", "video/webm"),
    ],
)
def test_sniff_by_signature(data: bytes, kind: str, mime: str) -> None:
    ft = ingest.sniff(data)
    assert ft is not None
    assert (ft.kind, ft.mime) == (kind, mime)


def test_unsupported_and_wrong_kind_are_415() -> None:
    with pytest.raises(ingest.RejectedUploadError) as exc:
        ingest.check_type(b"GIF89a....", "photo")
    assert exc.value.status == 415
    with pytest.raises(ingest.RejectedUploadError, match="use the photo upload"):
        ingest.check_type(_jpeg((8, 8)), "video")


def test_limits_reject_over_limit_files() -> None:
    s = Settings()
    ingest.check_size(s.photo_max_bytes, "photo", s)
    with pytest.raises(ingest.RejectedUploadError) as exc:
        ingest.check_size(s.photo_max_bytes + 1, "photo", s)
    assert exc.value.status == 413
    with pytest.raises(ingest.RejectedUploadError):
        ingest.check_size(s.video_max_bytes + 1, "video", s)
    # 8000 x 5001 is just over 40 MP; only the header is read.
    with pytest.raises(ingest.RejectedUploadError, match="megapixels"):
        ingest.check_photo_pixels(_png((8000, 5001)), s)
    assert ingest.check_photo_pixels(_png((8000, 5000)), s) == (8000, 5000)
    with pytest.raises(ingest.RejectedUploadError, match="Trim the clip to 10 seconds"):
        ingest.check_duration(12.0, s)
    ingest.check_duration(10.0, s)


def test_probe_reads_duration_without_decoding(tmp_path: Path) -> None:
    pytest.importorskip("av")
    path = tmp_path / "c.mp4"
    _clip(path, 4)
    seconds, w, h = ingest.probe_video(path)
    assert 3.5 <= seconds <= 4.5
    assert (w, h) == (160, 120)


@pytest.mark.parametrize(
    ("name", "safe"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\cat.jpg", "cat.jpg"),
        ("cat\x00\n.jpg", "cat.jpg"),
        ("<script>.png", "_script_.png"),
        ("", "upload"),
        (None, "upload"),
    ],
)
def test_safe_name(name: str | None, safe: str) -> None:
    assert ingest.safe_name(name) == safe


# --- API (needs Postgres) ---------------------------------------------------------


def _embed(crops: list[Image.Image]) -> np.ndarray:
    """Fake embedder: the crop's mean colour, repeated to the real embedding size."""
    means = [np.asarray(c, dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255 for c in crops]
    return np.stack([np.resize(m, EMBEDDING_DIM) for m in means]).astype(np.float32)


def _detector(img: Image.Image) -> list[Detection]:
    """Finds the red square (a 'cat'), wherever it is."""
    arr = np.asarray(img.convert("RGB"), dtype=int)
    mask = (arr[..., 0] > 180) & (arr[..., 1] < 90)
    if not mask.any():
        return []
    ys, xs = np.nonzero(mask)
    return [
        Detection("cat", 0.9, (float(xs.min()), float(ys.min()), xs.max() + 1.0, ys.max() + 1.0))
    ]


def _analyzer() -> Analyzer:
    rng = np.random.default_rng(0)
    classes = ["bengal", "domestic_cat", "ragdoll"]
    colours = [(40, 40, 220), (40, 200, 40), RED]
    xs = np.concatenate(
        [np.resize(np.array(c) / 255, (40, EMBEDDING_DIM)) for c in colours]
    ) + rng.normal(scale=0.01, size=(120, EMBEDDING_DIM))
    ys = np.repeat(np.arange(3), 40)
    head = train_species_head("cat", classes, xs, ys, xs, ys, xs, ys, kinds=("logreg",))
    cfg = PolicyConfig(
        species={
            "cat": SpeciesThresholds(
                min_confidence=0.5, max_knn_distance=0.5, review_knn_distance=0.9
            )
        }
    )
    return Analyzer(_detector, _embed, {"cat": head}, {"cat": KnnIndex(xs)}, cfg, version="test")


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set")
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    engine.dispose()
    migrate.upgrade(DATABASE_URL)
    settings = Settings(database_url=DATABASE_URL, data_dir=tmp_path)
    analyzer = _analyzer()
    with TestClient(create_app(settings, lambda: analyzer)) as c:
        yield c


def _scalar(sql: str) -> Any:
    assert DATABASE_URL is not None
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        value = conn.execute(text(sql)).scalar()
    engine.dispose()
    return value


def _cat_photo(size: tuple[int, int] = (640, 480)) -> bytes:
    img = Image.new("RGB", size, (128, 128, 128))
    img.paste(RED, (200, 150, 400, 350))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


@pytest.mark.db
def test_predict_image_stores_everything(client: TestClient, tmp_path: Path) -> None:
    r = client.post("/predict/image", files={"file": ("../my cat.jpg", _cat_photo())})
    assert r.status_code == 200, r.text
    run = r.json()
    assert run["upload"]["status"] == "done"
    assert run["upload"]["original_name"] == "my cat.jpg"
    assert len(run["frames"]) == 1 and len(run["detections"]) == 1
    pred = run["predictions"][0]
    assert (pred["outcome"], pred["label"]) == ("accept", "ragdoll")
    stored = Path(run["frames"][0]["path"])
    assert stored.is_relative_to(tmp_path) and stored.suffix == ".webp"
    with Image.open(stored) as im:
        assert im.size == (640, 480) and not im.info.get("exif")
    assert Path(run["detections"][0]["crop_path"]).stat().st_size <= 40_000
    assert _scalar("SELECT count(*) FROM detection WHERE embedding IS NOT NULL") == 1
    assert client.get(f"/runs/{run['upload']['id']}").json() == run


@pytest.mark.db
def test_exif_orientation_applied_once_and_gps_stripped(client: TestClient) -> None:
    img = Image.new("RGB", (640, 480), (128, 128, 128))
    img.paste(RED, (200, 150, 400, 350))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    exif[0x8825] = {2: (48.0, 51.0, 24.0)}  # GPS latitude
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    r = client.post("/predict/image", files={"file": ("phone.jpg", buf.getvalue())})
    assert r.status_code == 200, r.text
    frame = r.json()["frames"][0]
    assert (frame["width"], frame["height"]) == (480, 640)
    with Image.open(frame["path"]) as stored:
        assert not stored.getexif()


@pytest.mark.db
def test_forty_megapixel_photo_is_stored_under_250_kb(client: TestClient) -> None:
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (1250, 2000, 3), dtype=np.uint8)
    img = Image.fromarray(noise).resize((8000, 5000), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    assert len(buf.getvalue()) < 15 * 2**20
    r = client.post("/predict/image", files={"file": ("big.jpg", buf.getvalue())})
    assert r.status_code == 200, r.text
    frame = r.json()["frames"][0]
    assert frame["size_bytes"] <= 250_000
    assert max(frame["width"], frame["height"]) <= 1024


@pytest.mark.db
@pytest.mark.parametrize(
    ("endpoint", "name", "data", "status", "message"),
    [
        ("/predict/image", "big.jpg", b"\xff\xd8\xff" + b"\0" * (15 * 2**20), 413, "15 MB"),
        ("/predict/image", "huge.png", _png((8000, 5001)), 413, "megapixels"),
        ("/predict/image", "fake.jpg", b"MZ\x90\x00 not an image", 415, "Unsupported"),
        ("/predict/video", "big.mp4", b"\0\0\0\x18ftypisom" + b"\0" * (25 * 2**20), 413, "25 MB"),
        ("/predict/video", "photo.mp4", _jpeg((8, 8)), 415, "use the photo upload"),
    ],
)
def test_over_limit_uploads_are_rejected_and_logged(
    client: TestClient, endpoint: str, name: str, data: bytes, status: int, message: str
) -> None:
    r = client.post(endpoint, files={"file": (name, data)})
    assert r.status_code == status
    assert message in r.json()["detail"]
    assert _scalar("SELECT count(*) FROM dead_letter") == 1
    assert _scalar("SELECT count(*) FROM upload") == 0


@pytest.mark.db
def test_long_clip_is_rejected_with_trim_message(client: TestClient, tmp_path: Path) -> None:
    pytest.importorskip("av")
    data = _clip(tmp_path / "long.mp4", 12)
    r = client.post("/predict/video", files={"file": ("long.mp4", data)})
    assert r.status_code == 413
    assert "Trim the clip to 10 seconds or upload photos instead" in r.json()["detail"]
    assert _scalar("SELECT reason FROM dead_letter").startswith("Clip is 12")
    assert not any((tmp_path / "inbox").iterdir())  # the temporary file is gone


@pytest.mark.db
def test_predict_video_groups_one_cat(client: TestClient, tmp_path: Path) -> None:
    pytest.importorskip("av")
    data = _clip(tmp_path / "cat.mp4", 4)
    r = client.post("/predict/video", files={"file": ("cat.mp4", data)})
    assert r.status_code == 200, r.text
    run = r.json()
    assert len(run["frames"]) == 4
    assert len(run["predictions"]) == 1
    assert run["predictions"][0]["group_idx"] == 0
    assert run["predictions"][0]["label"] == "ragdoll"
    assert not any((tmp_path / "inbox").iterdir())


@pytest.mark.db
def test_unknown_ids_are_404(client: TestClient) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/runs/{missing}").status_code == 404
    assert client.get(f"/jobs/{missing}").status_code == 404
    assert client.get("/health").json() == {"status": "ok"}


def test_storage_report_counts_files(tmp_path: Path) -> None:
    from petlens.storage import report

    (tmp_path / "uploads" / "frames").mkdir(parents=True)
    (tmp_path / "uploads" / "frames" / "a.webp").write_bytes(b"x" * 2048)
    text_out = report(Settings(data_dir=tmp_path))
    assert "uploads/frames" in text_out and "1" in text_out


def _seed_index(tmp_path: Path) -> list[str]:
    """Six ragdoll and three bengal training photos, with stored images for the ragdolls."""
    from petlens.data.imaging import stored_path
    from petlens.db.evidence import TrainRecord, replace_index
    from petlens.db.session import session_scope

    rng = np.random.default_rng(1)
    records, ragdolls = [], []
    for i in range(9):
        breed, colour = ("ragdoll", RED) if i < 6 else ("bengal", (40, 40, 220))
        vec = np.resize(np.array(colour) / 255, EMBEDDING_DIM) + rng.normal(
            0, 0.01 * i, EMBEDDING_DIM
        )
        image_id = f"{i:064x}"
        records.append(
            TrainRecord(
                image_id,
                "cat",
                breed,
                "oxford",
                f"ref{i}",
                "CC-BY-SA-4.0",
                "someone",
                [float(v) for v in vec],
            )
        )
        if breed == "ragdoll":
            ragdolls.append(image_id)
            path = stored_path(tmp_path / "processed" / "images", image_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), RED).save(path, "WEBP")
    with session_scope(DATABASE_URL) as session:
        assert replace_index(session, records, "v-test", "fake") == 9
    return ragdolls


@pytest.mark.db
def test_evidence_shows_nearest_photos_of_the_suggested_breed(
    client: TestClient, tmp_path: Path
) -> None:
    ragdolls = _seed_index(tmp_path)
    run = client.post("/predict/image", files={"file": ("cat.jpg", _cat_photo())}).json()
    pred = run["predictions"][0]
    evidence = run["evidence"][pred["id"]]
    assert len(evidence) == 5
    assert all(e["breed"] == "ragdoll" for e in evidence)
    sims = [e["similarity"] for e in evidence]
    assert sims == sorted(sims, reverse=True)
    assert evidence[0]["image_id"] == ragdolls[0]  # the least noisy copy is the closest
    assert evidence[0]["licence"] == "CC-BY-SA-4.0"
    r = client.get(evidence[0]["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    frame = client.get(f"/images/frames/{run['frames'][0]['image_id']}")
    assert frame.status_code == 200


@pytest.mark.db
def test_group_evidence_for_clips(client: TestClient, tmp_path: Path) -> None:
    pytest.importorskip("av")
    _seed_index(tmp_path)
    run = client.post(
        "/predict/video", files={"file": ("cat.mp4", _clip(tmp_path / "c.mp4", 3))}
    ).json()
    assert all(d["group_idx"] == 0 for d in run["detections"])
    pred = run["predictions"][0]
    assert [e["breed"] for e in run["evidence"][pred["id"]]] == ["ragdoll"] * 5


@pytest.mark.db
@pytest.mark.parametrize(
    "url", ["/images/train/" + "0" * 63, "/images/secrets/" + "0" * 64, "/images/train/..%2F.env"]
)
def test_image_route_only_serves_known_hashes(client: TestClient, url: str) -> None:
    assert client.get(url).status_code == 404


@pytest.mark.db
def test_jobs_list_and_limits(client: TestClient, tmp_path: Path) -> None:
    folder = tmp_path / "batch"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(_cat_photo())
    r = client.post("/jobs/review-batch", json={"folder": str(folder)})
    assert r.status_code == 202
    assert r.json()["progress_total"] == 1
    jobs = client.get("/jobs").json()
    assert [j["id"] for j in jobs] == [r.json()["id"]]
    assert (
        client.post("/jobs/review-batch", json={"folder": str(tmp_path / "nope")}).status_code
        == 422
    )
    assert client.get("/limits").json()["video_max_seconds"] == 10.0
