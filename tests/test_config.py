from pathlib import Path

import pytest

from petlens.config import Settings


def test_defaults_point_at_local_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    settings = Settings(_env_file=None)
    assert settings.database_url.startswith("postgresql+psycopg://")
    assert "localhost:5432" in settings.database_url


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@db:5432/x")
    settings = Settings(_env_file=None)
    assert settings.data_dir == tmp_path
    assert settings.database_url.endswith("@db:5432/x")


def test_data_subfolders(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    assert settings.raw_dir == tmp_path / "raw"
    assert settings.processed_dir == tmp_path / "processed"
    assert settings.models_dir == tmp_path / "models"
    assert settings.footage_dir == tmp_path / "footage"
