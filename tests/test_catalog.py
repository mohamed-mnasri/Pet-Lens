import pytest
from pydantic import ValidationError

from petlens.data.catalog import BreedCatalog, load_breeds, load_sources


def test_breeds_yaml_is_valid_and_complete() -> None:
    cat = load_breeds()
    assert len(cat.by_oxford_label()) == 37
    cats = cat.training_classes("cat")
    dogs = cat.training_classes("dog")
    assert len(cats) == 25 and "domestic_cat" in cats
    assert len(dogs) == 25
    unknown = {b.id for b in cat.breeds if b.role == "unknown"}
    assert unknown == {"chartreux", "turkish_van", "korat", "selkirk_rex", "manx"}
    # Unknown breeds are never Oxford breeds (Oxford breeds all stay in training).
    assert all(cat.get(u).oxford_label is None for u in unknown)


def test_every_cat_has_commons_categories_and_dogs_do_not_yet() -> None:
    cat = load_breeds()
    for b in cat.breeds:
        assert bool(b.commons_categories) == (b.species == "cat"), b.id


def test_duplicate_ids_rejected() -> None:
    b = {"id": "x", "name": "X", "species": "cat", "role": "train"}
    with pytest.raises(ValidationError, match="duplicate breed ids"):
        BreedCatalog.model_validate({"breeds": [b, b]})


def test_close_pair_must_reference_known_breeds() -> None:
    b = {"id": "x", "name": "X", "species": "cat", "role": "train"}
    with pytest.raises(ValidationError, match="unknown breed"):
        BreedCatalog.model_validate({"breeds": [b], "close_pairs": [["x", "y"]]})


def test_sources_yaml() -> None:
    src = load_sources().sources
    ox = src["oxford"]
    assert ox.licence == "CC BY-SA 4.0"
    assert {f.name for f in ox.files} == {"images.tar.gz", "annotations.tar.gz"}
    assert set(src["commons"].licence_allowlist) == {"CC0", "Public domain", "CC BY", "CC BY-SA"}
