"""Application configuration loaded from environment variables.

Single source of truth for runtime settings. All other modules import from here
rather than reading ``os.environ`` directly — this keeps the config schema
testable and the production / docker-compose / pytest setups consistent.
"""

from __future__ import annotations

from functools import lru_cache

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
