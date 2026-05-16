#!/usr/bin/env sh
# API entrypoint: apply migrations then exec whatever command was passed.
# Migrations are idempotent; running them every cold start is intentional so
# that an image-rebuild-with-new-migrations gets applied automatically.

set -eu

echo "[entrypoint-api] applying alembic migrations"
alembic upgrade head

echo "[entrypoint-api] handing off to: $*"
exec "$@"
