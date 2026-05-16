"""Alembic migration environment.

The URL is sourced from ``factory_anomaly.config.DatabaseSettings`` instead of
``alembic.ini``, so the same env file works in local docker-compose, CI
(testcontainers), and production. ``alembic.ini`` keeps a placeholder URL only
so that Alembic's CLI doesn't error before we override it.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from factory_anomaly.config import get_database_settings
from factory_anomaly.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# URL resolution order:
#   1. ``-x url=...`` on the alembic CLI.
#   2. Whatever the caller (e.g. test fixture) already set on the Config.
#   3. ``factory_anomaly.config.DatabaseSettings`` (env vars).
# The alembic.ini placeholder ("driver://overridden-by-env-py") counts as
# unset and falls through to step 3.
existing_url = config.get_main_option("sqlalchemy.url") or ""
override_url = context.get_x_argument(as_dictionary=True).get("url")
if override_url:
    config.set_main_option("sqlalchemy.url", override_url)
elif not existing_url or existing_url.startswith("driver://"):
    config.set_main_option("sqlalchemy.url", get_database_settings().url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL without a live DB connection."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
