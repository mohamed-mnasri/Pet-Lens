"""Detection pipeline, people blurring, clip grouping and frame sampling (no real models)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from petlens.models.heads import train_species_head
from petlens.models.openset import KnnIndex
from petlens.models.policy import PolicyConfig, SpeciesThresholds
from petlens.vision.detect import Detection, class_names_of, merge_animals
from petlens.vision.grouping import GroupingConfig, Item, group_items
from petlens.vision.pipeline import Analyzer, analyse_clip, analyse_image
from petlens.vision.privacy import blur_people, head_box
from petlens.vision.video import Frame, sample_frames

CLASSES = ["bengal", "domestic_cat", "ragdoll"]
COLOURS = {"ragdoll": (220, 40, 40), "bengal": (40, 40, 220), "domestic_cat": (40, 200, 40)}


def _embed(crops: list[Image.Image]) -> np.ndarray:
    """Fake embedder: the crop's mean colour, which identifies our painted 'cats'."""
    out = [np.asarray(c, dtype=np.float32).reshape(-1, 3).mean(axis=0) / 255 for c in crops]
    return np.stack(out).astype(np.float32)


def _analyzer(detector) -> Analyzer:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    x, y = [], []
    for i, c in enumerate(CLASSES):
        base = np.array(COLOURS[c]) / 255
        x.append(base + rng.normal(scale=0.03, size=(40, 3)))
        y += [i] * 40
    xs, ys = np.concatenate(x), np.array(y)
    head = train_species_head("cat", CLASSES, xs, ys, xs, ys, xs, ys, kinds=("logreg",))
    cfg = PolicyConfig(
        species={
            "cat": SpeciesThresholds(
                min_confidence=0.5, max_knn_distance=0.5, review_knn_distance=0.9
            )
        }
    )
    return Analyzer(
        detector,
        _embed,
        {"cat": head},
        {"cat": KnnIndex(xs)},
        cfg,
        grouping=GroupingConfig(min_similarity=0.9),
    )


def _scene(
    cats: list[tuple[str, tuple[int, int]]], person: tuple[int, int] | None = None
) -> tuple[Image.Image, list[Detection]]:
    """Grey background with painted squares; returns the image and its 'true' detections."""
    img = Image.new("RGB", (320, 240), (128, 128, 128))
    dets: list[Detection] = []
    for breed, (x, y) in cats:
        img.paste(COLOURS[breed], (x, y, x + 60, y + 60))
        dets.append(Detection("cat", 0.9, (x, y, x + 60, y + 60)))
    if person:
        x, y = person
        for i in range(0, 50, 2):  # stripes, so a blur visibly changes them
            img.paste((0, 0, 0) if i % 4 else (255, 255, 255), (x, y + i, x + 50, y + i + 2))
        dets.append(Detection("person", 0.8, (x, y, x + 50, y + 50)))
    return img, dets


class FakeDetector:
    def __init__(self) -> None:
        self.answers: dict[int, list[Detection]] = {}

    def add(self, img: Image.Image, dets: list[Detection]) -> Image.Image:
        self.answers[id(img)] = dets
        return img

    def __call__(self, img: Image.Image) -> list[Detection]:
        return self.answers.get(id(img), [])


def _frames(det: FakeDetector, scenes: list) -> list[Frame]:  # type: ignore[type-arg]
    frames = []
    for i, (img, dets) in enumerate(scenes):
        img = img.convert("RGB")  # already RGB, so analyse_image keeps the same object
        det.add(img, dets)
        frames.append(Frame(i, float(i), img))
    return frames


def test_photo_classifies_each_animal_and_blurs_people() -> None:
    det = FakeDetector()
    img, dets = _scene([("ragdoll", (20, 20)), ("bengal", (200, 100))], person=(120, 150))
    det.add(img, dets)
    res = analyse_image(_analyzer(det), img)
    labels = sorted(a.decision.label or "" for a in res.animals)
    assert labels == ["bengal", "ragdoll"]
    assert all(a.decision.outcome == "accept" for a in res.animals)
    assert res.people_blurred == 1
    before = np.asarray(img.crop((120, 150, 170, 170)), dtype=float)
    after = np.asarray(res.image.crop((120, 150, 170, 170)), dtype=float)
    assert after.std() < before.std() / 3  # head region smeared out
    assert res.image.getpixel((30, 30)) == COLOURS["ragdoll"]  # animals untouched


def test_one_cat_clip_gives_one_answer() -> None:
    det = FakeDetector()
    scenes = [_scene([("ragdoll", (20 + 10 * i, 30))]) for i in range(5)]
    clip = analyse_clip(_analyzer(det), _frames(det, scenes))
    assert not clip.ambiguous
    assert len(clip.groups) == 1
    assert clip.groups[0].frames == [0, 1, 2, 3, 4]
    assert clip.groups[0].decision.label == "ragdoll"


def test_two_cat_clip_gives_two_answers() -> None:
    det = FakeDetector()
    scenes = [
        _scene([("ragdoll", (20, 30 + 5 * i)), ("bengal", (220, 120 - 5 * i))]) for i in range(4)
    ]
    clip = analyse_clip(_analyzer(det), _frames(det, scenes))
    assert not clip.ambiguous
    assert sorted(g.decision.label or "" for g in clip.groups) == ["bengal", "ragdoll"]
    assert all(len(g.frames) == 4 for g in clip.groups)


def test_two_identical_looking_cats_make_grouping_ambiguous() -> None:
    det = FakeDetector()
    scenes = [
        _scene([("ragdoll", (20, 30)), ("ragdoll", (200, 30))]),
        _scene([("ragdoll", (110, 30))]),
    ]
    clip = analyse_clip(_analyzer(det), _frames(det, scenes))
    assert clip.ambiguous and clip.reason and "equally like two animals" in clip.reason
    assert clip.groups == []
    assert sum(len(f.animals) for f in clip.frames) == 3  # per-frame results still there


def test_grouping_never_merges_within_a_frame_and_checks_size() -> None:
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    items = [
        Item(0, "cat", v, 100.0),
        Item(0, "cat", v, 100.0),  # same frame: must stay separate
        Item(1, "cat", v, 10_000.0),  # 100x bigger: not the same animal
        Item(1, "dog", v, 100.0),  # other species
    ]
    res = group_items(items, GroupingConfig(ambiguity_margin=0.0))
    assert len(res.groups) == 4


def test_blur_covers_only_the_head() -> None:
    img = Image.new("RGB", (100, 200), (128, 128, 128))
    for i in range(0, 200, 2):
        img.paste((0, 0, 0) if i % 4 else (255, 255, 255), (0, i, 100, i + 2))
    # A standing person 40 px wide and 180 px tall: the head is the top 24 px.
    assert head_box((30, 10, 70, 190)) == pytest.approx((30, 10, 70, 34))
    out = blur_people(img, [Detection("person", 0.9, (30, 10, 70, 190))], pad=0)
    head = np.asarray(out.crop((32, 12, 68, 32)), dtype=float)
    body = (40, 100, 60, 180)  # where a held cat would be
    assert head.std() < np.asarray(img.crop((32, 12, 68, 32)), dtype=float).std() / 3
    assert np.array_equal(np.asarray(out.crop(body)), np.asarray(img.crop(body)))


def test_blur_ignores_animals() -> None:
    img = Image.new("RGB", (100, 100), (10, 200, 10))
    out = blur_people(img, [Detection("cat", 0.9, (0, 0, 50, 50))])
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_sample_frames_one_per_second(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    path = tmp_path / "clip.mp4"
    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(35):  # 3.5 seconds at 10 fps
            frame = av.VideoFrame.from_image(Image.new("RGB", (64, 48), (i * 7, 0, 0)))
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    frames = sample_frames(path)
    assert [round(f.seconds) for f in frames] == [0, 1, 2, 3]
    assert frames[0].image.size == (64, 48)
    assert len(sample_frames(path, max_frames=2)) == 2


class _Found:
    def __init__(self, class_id: list[int], data: dict[str, object] | None = None) -> None:
        self.class_id = np.asarray(class_id)
        self.data = data or {}


def test_class_names_use_coco_ids_not_list_index() -> None:
    # RF-DETR puts COCO category ids in class_id (cat 17, dog 18, person 1).
    assert class_names_of(_Found([17, 18, 1, 62])) == ["cat", "dog", "person", ""]
    # When the prediction carries names, those win.
    found = _Found([17], {"class_name": np.asarray(["cat"])})
    assert class_names_of(found) == ["cat"]


def test_merge_animals_keeps_one_box_per_animal_with_alternatives() -> None:
    box = (100.0, 100.0, 300.0, 400.0)
    near = (102.0, 99.0, 298.0, 398.0)
    far = (500.0, 100.0, 650.0, 300.0)
    merged = merge_animals(
        [
            Detection("dog", 0.50, near),
            Detection("cat", 0.60, box),
            Detection("dog", 0.21, box),  # too weak to count as an alternative
            Detection("cat", 0.90, far),
            Detection("cat", 0.40, (700.0, 0.0, 760.0, 60.0)),  # alone and below threshold
        ]
    )
    assert [(d.label, d.score, d.alternatives) for d in merged] == [
        ("cat", 0.90, ()),
        ("cat", 0.60, ("dog",)),
    ]
    # The detector's top label can be wrong: a weaker "cat" on the box is kept as an option.
    flipped = merge_animals([Detection("dog", 0.56, box), Detection("cat", 0.35, near)])
    assert [(d.label, d.alternatives) for d in flipped] == [("dog", ("cat",))]


def test_species_from_embedding_when_detector_is_unsure() -> None:
    img, dets = _scene([("ragdoll", (40, 40))])
    unsure = [Detection("dog", 0.56, dets[0].box, alternatives=("cat",))]
    det = FakeDetector()
    det.add(img, unsure)
    res = analyse_image(_analyzer(det), img)
    assert [a.species for a in res.animals] == ["cat"]
    assert res.animals[0].decision.label == "ragdoll"
