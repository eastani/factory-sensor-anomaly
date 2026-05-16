# ADR-0002: Core tech stack & ML approach

- **Status:** Accepted
- **Date:** 2026-05-16
- **Deciders:** Naoya Higashitani

## Context

The stack must satisfy three goals: (a) demonstrate modern Python tooling that a senior reviewer would recognise as up-to-date, (b) keep the local feedback loop fast, (c) cleanly map to managed cloud services in Phase 2 without rewrite.

## Decision

### Language & tooling
- **Python 3.12** (3.13 still has uneven wheel support for `scipy`/`numpy` on some platforms as of mid-2026; 3.12 is the safe production target).
- **uv** for dependency management and virtual environment. Replaces `pip` + `pip-tools` + `virtualenv`.
- **Ruff** as the single linter *and* formatter. Replaces Black + Flake8 + isort.
- **mypy** in `strict` mode from day one — easier to maintain than to retrofit.
- **pytest** + **pytest-cov** with `--cov-fail-under=70` enforced in CI.
- **pre-commit** with Ruff, gitleaks (secret scanning), and basic hygiene hooks.

### Application layers
- **FastAPI** with Pydantic v2 for the API.
- **SQLAlchemy 2.x** (typed) + **Alembic** for migrations. Migrations are mandatory from Phase 1 — never `create_all`.
- **PostgreSQL 16** for sensor data and inference results.
- **Streamlit** + **Plotly** for the dashboard. Refresh via `st.autorefresh` (NOT a `while True: time.sleep` loop).
- **Docker** + **docker-compose** to wire the three services locally.

### ML approach
- **Baseline:** **Isolation Forest** on rolling-window features (mean, std, min, max per channel).
- **Improvement:** Apply **STL decomposition** to seasonal channels first, run IF on the residual. This addresses a known IF pitfall — periodic peaks (shift changes, scheduled cycles) get flagged as anomalies, and real anomalies coinciding with seasonal peaks get hidden. Pattern adopted from [Jacer7/Anomaly_Detection](https://github.com/Jacer7/Anomaly_Detection) and [Mishra (Medium, 2023)](https://medium.com/@richa.mishr01/anomaly-detection-in-seasonal-time-series-where-anomalies-coincide-with-seasonal-peaks-9859a6a6b8ba).
- **Persistence:** `joblib.dump` (more efficient than `pickle` for fitted numpy-heavy estimators per the [scikit-learn docs](https://scikit-learn.org/1.3/model_persistence.html)). Model artefacts include a metadata sidecar with `sklearn_version`, `python_version`, `training_data_hash`, and `trained_at`. A mismatched sklearn version on load is a hard failure, not a warning.

### Cloud (deferred)
The choice between **AWS (ECR + App Runner + RDS)** and **Azure (ACR + Container Apps + Azure DB for PostgreSQL)** is deferred to Phase 2. Both will be evaluated; for a personal project, Azure Container Apps' scale-to-zero is a meaningful cost advantage over App Runner. Infrastructure will be defined with **Terraform** from the start — no click-ops.

GitHub Actions will authenticate to the chosen cloud via **OIDC** (federated identity). Static access keys in GitHub Secrets are explicitly rejected.

## Alternatives considered

| Option | Why not |
|--------|---------|
| Poetry / Rye / pipenv | uv is materially faster (10–100×), single binary, actively developed by Astral, has become the de-facto modern choice. |
| Black + Flake8 + isort | Three tools, three configs. Ruff covers all of it and is faster. |
| Plain `create_all` for schema | Demoable once, painful forever. Alembic from Phase 1. |
| Dash / Gradio | Streamlit is the canonical choice for this stack and matches the user's stated plan. |
| Prophet / LSTM autoencoder | Possible Phase 1.5/3 upgrades; overkill for the baseline. |
| Celery + Redis (for Phase 3) | Heavyweight for a personal project. SQS / Azure Service Bus + a worker container is preferred — also more cloud-native and shows different architectural muscle. |

## Consequences

**Positive**
- Every tool choice has a written justification — easy to defend in interviews.
- mypy strict + Ruff strict means contributions stay clean automatically.
- Alembic + Terraform mean Phase 2 is "wire it up", not "rewrite for prod".

**Negative**
- mypy strict on day one slows the first few PRs. Acceptable tax.
- uv is newer than pip; some IDE integrations are not perfect (acceptable — CLI is the source of truth).
- Two ML "modes" (raw IF and STL+IF) double the test surface. Mitigated by parametrised pytest fixtures.
