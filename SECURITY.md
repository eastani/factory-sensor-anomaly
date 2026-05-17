# Security Policy

This is a personal portfolio project, not a production service. It is published to demonstrate engineering practice. The policies below are nonetheless real — the repo is set up to enforce them.

## Supported versions

Only `main` is supported. There are no tagged releases yet.

## What is in scope

- The application code under `src/`, `dashboard/`, `scripts/`, and `tests/`.
- Container images built from `Dockerfile`.
- Terraform under `infra/aws/` and `infra/azure/`.

## What is **not** in scope

- The synthetic and public-dataset CSVs under `data/` (no real customer data).
- Cloud infrastructure is **not** persistently deployed. AWS app stack is created on demand via `deploy-aws.yml` and torn down after evaluation. Only the OIDC / state / ECR bootstrap stack remains live.

## Secrets handling

- **Pre-commit `gitleaks` runs on every commit** (see `.pre-commit-config.yaml`).
- `.env` is git-ignored; `.env.example` is the only checked-in template.
- GitHub Actions deploy uses **OIDC + short-lived role assumption**, not long-lived AWS access keys.
- No real credentials should ever be in this repo. If you find one, report it via the channel below — do not open a public issue.

## Reporting a vulnerability

Email **taquanta777@gmail.com** with:

1. The commit SHA you found it on.
2. A description of the issue and reproduction steps.
3. The impact you believe it has.

I will acknowledge within 7 days. Because this is a single-maintainer portfolio repo, fix timelines are best-effort, but anything affecting deployed infrastructure or leaked credentials will be prioritised.

Please do **not** open a public GitHub issue for security reports.
