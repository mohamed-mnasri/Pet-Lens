"""Load and validate the label catalogue (``breeds.yaml``) and ``sources.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_DATA_DIR = Path(__file__).resolve().parents[3] / "data"

Species = Literal["cat", "dog"]
Role = Literal["train", "unknown"]


class Breed(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    name: str
    species: Species
    role: Role
    oxford_label: str | None = None
    commons_categories: list[str] = Field(default_factory=list)


class BreedCatalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    breeds: list[Breed]
    close_pairs: list[tuple[str, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_and_pairs(self) -> BreedCatalog:
        ids = [b.id for b in self.breeds]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate breed ids: {sorted(dupes)}")
        oxford = [b.oxford_label for b in self.breeds if b.oxford_label]
        if len(oxford) != len(set(oxford)):
            raise ValueError("duplicate oxford_label")
        known = set(ids)
        for a, b in self.close_pairs:
            if a not in known or b not in known:
                raise ValueError(f"close pair uses unknown breed: {a}, {b}")
        return self

    def get(self, breed_id: str) -> Breed:
        for b in self.breeds:
            if b.id == breed_id:
                return b
        raise KeyError(breed_id)

    def by_oxford_label(self) -> dict[str, Breed]:
        return {b.oxford_label: b for b in self.breeds if b.oxford_label}

    def training_classes(self, species: Species) -> list[str]:
        """Class ids the per-species head is trained on, in a stable order."""
        return sorted(b.id for b in self.breeds if b.species == species and b.role == "train")


class SourceFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")
    urls: list[str] = Field(min_length=1)


class Source(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    homepage: str
    licence: str
    attribution: str
    files: list[SourceFile] = Field(default_factory=list)
    licence_allowlist: list[str] = Field(default_factory=list)


class SourceCatalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sources: dict[str, Source]


def _read_yaml(path: Path) -> object:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_breeds(path: Path | None = None) -> BreedCatalog:
    return BreedCatalog.model_validate(_read_yaml(path or REPO_DATA_DIR / "breeds.yaml"))


def load_sources(path: Path | None = None) -> SourceCatalog:
    return SourceCatalog.model_validate(_read_yaml(path or REPO_DATA_DIR / "sources.yaml"))
