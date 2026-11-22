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
    # Datasets, weights and footage. Set DATA_DIR in .env (D:/petlens-data on my machine).
    data_dir: Path = Path(".petlens-data")

    # API (Step 6). Local only: bind to loopback.
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    detector_size: str = "base"

    # Upload limits, checked before anything is decoded or stored (PLAN.md Step 6).
    photo_max_bytes: int = 15 * 1024 * 1024
    photo_max_pixels: int = 40_000_000
    video_max_bytes: int = 25 * 1024 * 1024
    video_max_seconds: float = 10.0
    crop_long_edge: int = 384
    crop_max_bytes: int = 40_000

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

    @property
    def uploads_dir(self) -> Path:
        """Compressed, face-blurred copies of uploaded photos and sampled frames."""
        return self.data_dir / "uploads"

    @property
    def inbox_dir(self) -> Path:
        """Short-lived files while a clip is read; deleted straight after."""
        return self.data_dir / "inbox"


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
