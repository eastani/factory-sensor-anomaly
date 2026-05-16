.PHONY: help install lint format typecheck test check clean pre-commit-install pre-commit-run

help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install all dependencies including dev tools.
	uv sync --all-groups

lint:  ## Run Ruff lint and format check.
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-fix lint and format code.
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## Run mypy strict type-check.
	uv run mypy

test:  ## Run pytest with coverage.
	uv run pytest

check: lint typecheck test  ## Run all quality gates (lint + types + tests).

pre-commit-install:  ## Install pre-commit git hooks.
	uv run pre-commit install

pre-commit-run:  ## Run all pre-commit hooks on every file.
	uv run pre-commit run --all-files

clean:  ## Remove caches and build artifacts.
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
