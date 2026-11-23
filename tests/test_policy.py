"""Decision policy: every branch, tuning, open-set score and reporting."""

from pathlib import Path

import numpy as np
import pytest

from petlens.models import policy
from petlens.models.openset import KnnIndex
from petlens.models.policy import PolicyConfig, SpeciesThresholds, decide

CATS = ["bengal", "domestic_cat", "egyptian_mau", "ragdoll"]
PAIRS = [("bengal", "egyptian_mau")]
CFG = PolicyConfig(
    close_margin=0.15,
    species={
        "cat": SpeciesThresholds(min_confidence=0.6, max_knn_distance=0.4, review_knn_distance=0.7),
        "dog": SpeciesThresholds(min_confidence=0.6, max_knn_distance=0.4, review_knn_distance=0.7),
    },
)


def _p(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def test_accept_confident_and_familiar() -> None:
    d = decide(_p(0.05, 0.05, 0.05, 0.85), CATS, 0.2, "cat", CFG, PAIRS)
    assert (d.outcome, d.label, d.also) == ("accept", "ragdoll", None)
    assert d.top3[0][0] == "ragdoll"
    assert d.top3[0][1] == pytest.approx(0.85)


def test_accept_names_close_pair_partner() -> None:
    d = decide(_p(0.62, 0.0, 0.38, 0.0), CATS, 0.2, "cat", CFG, PAIRS)
    assert (d.outcome, d.label) == ("accept", "bengal")
    assert d.also is None  # margin 0.24 is wider than 0.15
    d = decide(_p(0.66, 0.0, 0.0, 0.34), CATS, 0.2, "cat", CFG, PAIRS)
    assert d.also is None  # ragdoll isn't bengal's close pair
    cfg = CFG.model_copy(update={"close_margin": 0.3})
    d = decide(_p(0.62, 0.0, 0.38, 0.0), CATS, 0.2, "cat", cfg, PAIRS)
    assert d.also == "egyptian_mau"


def test_domestic_wins_gives_no_clear_breed() -> None:
    d = decide(_p(0.1, 0.8, 0.05, 0.05), CATS, 0.2, "cat", CFG, PAIRS)
    assert (d.outcome, d.label) == ("no_clear_breed", "domestic_cat")


def test_low_confidence_cat_is_domestic_no_clear_breed() -> None:
    d = decide(_p(0.4, 0.2, 0.2, 0.2), CATS, 0.2, "cat", CFG, PAIRS)
    assert (d.outcome, d.label) == ("no_clear_breed", "domestic_cat")


def test_unfamiliar_but_not_odd_is_no_clear_breed() -> None:
    d = decide(_p(0.9, 0.0, 0.1, 0.0), CATS, 0.5, "cat", CFG, PAIRS)
    assert d.outcome == "no_clear_breed"


def test_far_from_training_goes_to_review() -> None:
    d = decide(_p(0.9, 0.0, 0.1, 0.0), CATS, 0.9, "cat", CFG, PAIRS)
    assert (d.outcome, d.label) == ("review", None)


def test_dog_without_clear_breed_has_no_label() -> None:
    d = decide(_p(0.5, 0.3, 0.2), ["beagle", "boxer", "pug"], 0.2, "dog", CFG, PAIRS)
    assert (d.outcome, d.label) == ("no_clear_breed", None)


def test_knn_distance_orders_familiar_before_unfamiliar() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 8)) + 5  # one tight region
    index = KnnIndex(train, k=5)
    near = index.distance(train[:3] + 0.01)
    far = index.distance(-train[:3])
    assert near.max() < 0.05 and far.min() > 1.0


def test_tune_meets_targets_on_the_tuning_slice() -> None:
    rng = np.random.default_rng(1)
    n = 400
    y = rng.integers(0, 4, size=n)
    conf = rng.uniform(0.3, 1.0, size=n)
    correct = rng.uniform(size=n) < conf  # higher confidence, more often right
    probs = np.full((n, 4), 0.0)
    pred = np.where(correct, y, (y + 1) % 4)
    probs[np.arange(n), pred] = conf
    for i in range(n):
        others = [c for c in range(4) if c != pred[i]]
        probs[i, others] = (1 - conf[i]) / 3
    knn = rng.uniform(0.1, 0.5, size=n)
    t = policy.tune(probs, y, knn, CATS, "cat", PolicyConfig(), PAIRS, review_knn_distance=0.6)
    cfg = PolicyConfig(species={"cat": t})
    ev = policy.evaluate(probs, y, knn, CATS, "cat", cfg, PAIRS)
    assert ev.selective_accuracy >= 0.90
    dom = y == 1
    claims = policy.evaluate(probs[dom], y[dom], knn[dom], CATS, "cat", cfg, PAIRS).accepted
    assert claims / dom.sum() <= 0.10
    assert ev.accepted > 0


def test_risk_coverage_falls_as_coverage_shrinks() -> None:
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
    y = np.array([0, 0, 1, 1])
    rc = policy.risk_coverage(probs, y, points=4)
    assert [p["risk"] for p in rc] == [0.0, 0.0, pytest.approx(1 / 3), 0.5]


def test_policy_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    policy.save(CFG, path)
    assert policy.load(path) == CFG
