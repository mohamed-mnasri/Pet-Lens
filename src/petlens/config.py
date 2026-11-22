"""Runtime settings, read from environment variables or a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Every field can be overridden with an environment variable of the same name
    (case-insensitive), e.g. ``DATABASE_URL`` or ``DATA_DIR``.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://petlens:petlens@localhost:5432/petlens"
    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def footage_dir(self) -> Path:
        return self.data_dir / "footage"


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
