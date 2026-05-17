"""Application configuration loaded from environment variables.

Single source of truth for runtime settings. All other modules import from here
rather than reading ``os.environ`` directly — this keeps the config schema
testable and the production / docker-compose / pytest setups consistent.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Postgres connection settings, populated from ``POSTGRES_*`` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    user: str = Field(default="anomaly")
    password: str = Field(default="changeme")
    db: str = Field(default="anomaly")
    host: str = Field(default="localhost")
    port: int = Field(default=5432)

    @property
    def url(self) -> str:
        """SQLAlchemy URL using psycopg v3 driver."""
        dsn = PostgresDsn.build(
            scheme="postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            path=self.db,
        )
        return str(dsn)


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    """Cached accessor so settings are parsed once per process."""
    return DatabaseSettings()


class ApiSettings(BaseSettings):
    """API-layer settings, populated from ``API_*`` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path = Field(default=Path("models/baseline.joblib"))
    window_size: int = Field(default=20, ge=2)
    log_level: str = Field(default="INFO")
    machine_id_default: str = Field(default="pump-001")


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    return ApiSettings()


class ImageApiSettings(BaseSettings):
    """Image-anomaly service settings (Phase 3.2; see ADR-0006).

    Populated from ``IMAGE_API_*`` env vars (note the prefix differs from
    the sensor API's ``API_*`` so the two configs can coexist in one
    ``.env`` without ambiguity).
    """

    model_config = SettingsConfigDict(
        env_prefix="IMAGE_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path = Field(default=Path("models/image_bank.joblib"))
    input_size: int = Field(default=224, ge=32)
    log_level: str = Field(default="INFO")
    # Cap CPU threads so this service does not starve the sensor stack when
    # both run in one docker-compose. Override via IMAGE_API_TORCH_NUM_THREADS.
    torch_num_threads: int = Field(default=2, ge=1)


@lru_cache(maxsize=1)
def get_image_api_settings() -> ImageApiSettings:
    return ImageApiSettings()
