# ADR-0005: Cloud architecture (AWS primary, Azure secondary)

- **Status:** Accepted
- **Date:** 2026-05-16
- **Deciders:** Naoya Higashitani

## Context

ADR-0002 left the AWS-vs-Azure choice open. Phase 2 forces a decision because
that's when the API actually has to run somewhere a recruiter can `curl`.

Two non-negotiables drove the design:

1. **No long-lived secrets in CI.** GitHub Actions authenticates to the cloud
   via OIDC federation. Static AWS access keys / Azure service principal
   passwords in GitHub Secrets are explicitly rejected.
2. **Cost ceiling.** This is a personal portfolio project. The deployed stack
   must be either inexpensive (< $50/month) or fully tear-down-able with one
   command. Both targets here are.

## Decision

Deploy **API only** to managed-container services in **both clouds**, with
AWS as the primary target (CI/CD active by default) and Azure as a sandbox.
Dashboard stays local for now — at $25/month per App Runner service, doubling
the bill for an internal-only UI is not worth it.

### AWS (primary)

| Component | Service | Why |
|---|---|---|
| Image registry | Amazon ECR (private) | Native integration with App Runner pull |
| Compute | AWS App Runner | Cheapest "deploy a container" with HTTPS + auto-scale built-in. No Kubernetes muscle to flex for one container. |
| Database | RDS for PostgreSQL `db.t4g.micro` Single-AZ | ~$13/month; matches the docker-compose Postgres 16 image. |
| Networking | VPC connector (App Runner → private RDS) | RDS stays private; App Runner reaches it over the VPC connector. Public RDS rejected — saves nothing and weakens the security posture. |
| Auth (CI → AWS) | IAM OIDC provider + role | GitHub Actions assumes an IAM role via the GitHub OIDC issuer. No keys. |
| State (Terraform) | S3 bucket + DynamoDB lock table | Industry-standard remote state. Bootstrapped once by a separate stack. |
| Cost guard | CloudWatch Billing Alarm | $10 / $30 / $50 thresholds. Tag every resource `Project=factory-sensor-anomaly`. |

### Azure (secondary)

| Component | Service | Why |
|---|---|---|
| Image registry | Azure Container Registry (Basic) | $5/month; trivial integration with Container Apps. |
| Compute | Azure Container Apps (Consumption) | **Scales to zero** — costs ~$0 when idle. Beats App Runner on idle cost. |
| Database | Azure DB for PostgreSQL Flexible Server, Burstable B1ms | ~$12/month. |
| Networking | VNet integration | Same private-DB-only stance as AWS. |
| Auth (CI → Azure) | Workload Identity Federation on App Registration | The Azure equivalent of OIDC. No client secrets. |
| State (Terraform) | Azure Storage Account container | Mirror of the S3 backend pattern. |

## Alternatives considered

| Option | Why rejected |
|---|---|
| ECS Fargate (AWS) | More Terraform code than App Runner for the same single-container workload. Saved for Phase 3 when ingester/scorer need orchestration. |
| Aurora Serverless v2 | Cheaper at very low utilisation, but baseline cost is similar to `db.t4g.micro` and the warmup cold-start adds operational complexity for no demo win. |
| Public RDS with security group on 0.0.0.0/0 | Slightly less Terraform code; meaningfully worse security posture. Rejected. |
| Deploying the Streamlit dashboard too | Doubles the always-on App Runner bill for an internal-only UI. Phase 4 will revisit with a public Grafana dashboard instead. |
| Static access keys in GitHub Secrets | Explicitly rejected; OIDC is mandatory. |

## Security model

1. **No root user usage** beyond the one-time IAM Identity Center / first
   IAM admin user creation. Root MFA is required; root has no access keys.
2. **OIDC trust policy** scopes the assumable role to:
   - Issuer: `https://token.actions.githubusercontent.com`
   - Subject: `repo:eastani/factory-sensor-anomaly:ref:refs/heads/main`
   No other branches can deploy.
3. **Least-privilege IAM policy** on the role: ECR push, App Runner update,
   RDS describe (no delete, no IAM, no billing).
4. **Image signing**: Phase 3 will add `cosign`-signed images and a
   verification step in the App Runner deployment job. Out of scope here.

## Tear-down

```bash
make cloud-down-aws        # cd infra/aws && terraform destroy
make cloud-down-azure      # cd infra/azure && terraform destroy
```

Tagged resources make orphan cleanup trivial via the AWS resource group or
Azure tag-search if Terraform state is ever lost.
