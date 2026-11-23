import pytest
from pydantic import ValidationError

from petlens.data.catalog import BreedCatalog, load_breeds, load_sources


def test_breeds_yaml_is_valid_and_complete() -> None:
    cat = load_breeds()
    assert len(cat.by_oxford_label()) == 37
    cats = cat.training_classes("cat")
    dogs = cat.training_classes("dog")
    assert len(cats) == 22 and "domestic_cat" in cats
    assert len(dogs) == 29
    unknown = {b.id for b in cat.breeds if b.role == "unknown"}
    assert unknown == {
        "bombay",
        "exotic_shorthair",
        "turkish_angora",
        "chartreux",
        "turkish_van",
        "korat",
        "selkirk_rex",
        "manx",
    }
    # Bombay is the one Oxford breed I chose not to cover; it tests unknown breeds.
    assert {u for u in unknown if cat.get(u).oxford_label} == {"bombay"}


def test_commons_categories_cover_cats_and_the_non_oxford_dogs() -> None:
    cat = load_breeds()
    for b in cat.breeds:
        wanted = b.species == "cat" or b.oxford_label is None
        assert bool(b.commons_categories) == wanted, b.id


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
