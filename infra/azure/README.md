# Azure deployment (Phase 2B)

Parallel of the AWS stack ([../aws/](../aws/)). Ships the same image to a
different cloud so the project can claim genuine multi-cloud experience.

## Why Azure was added as the secondary

- **Scale-to-zero is real** on Container Apps Consumption. Steady-state idle
  cost is ~$0 for compute (Postgres is the only meaningful line item).
- **Workload Identity Federation** is Azure's equivalent of AWS IAM OIDC —
  same security posture (no static credentials in CI), different vocabulary.

See [ADR-0005](../../docs/adr/0005-cloud-architecture.md) for the full
trade-off discussion.

## One-time setup

```bash
# Log in with the account that owns / has access to the subscription.
az login
az account show --query id -o tsv          # confirm the right subscription

# Bootstrap: state storage + app registration + federated credential.
cd infra/bootstrap-azure
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars (subscription_id, state_storage_account_name).
terraform init
terraform apply
```

Bootstrap outputs values you paste into GitHub Actions secrets:

| Secret name | Source |
|-------------|--------|
| `AZURE_TENANT_ID` | `terraform output tenant_id` |
| `AZURE_SUBSCRIPTION_ID` | `terraform output subscription_id` |
| `AZURE_CLIENT_ID` | `terraform output client_id` |
| `AZURE_TFSTATE_RG` | `terraform output shared_resource_group` |
| `AZURE_TFSTATE_ACCOUNT` | `terraform output state_storage_account` |
| `AZURE_TFSTATE_CONTAINER` | `terraform output state_container` |

## Deploy the application

After bootstrap and secrets, push to `main` (or trigger
`.github/workflows/deploy-azure.yml` manually). The workflow:

1. Authenticates via OIDC (no client secrets).
2. Builds the Docker image, pushes to ACR tagged with the commit SHA.
3. Runs `terraform apply` against `infra/azure/`.
4. Curls `/healthz` until Container Apps reports ready.

## Tear-down

```bash
cd infra/azure
terraform destroy
```

The bootstrap stack stays — destroying it would erase the federated
credential and state.

## Cost estimate

| Resource | Cost |
|----------|------|
| Container Apps Consumption | ~$0 when idle, scales with traffic |
| Azure DB for PostgreSQL Flexible (B1ms) | ~$12/month |
| ACR Basic | $5/month |
| Log Analytics | ~$0-2 (low volume) |
| **Total (idle)** | **~$17-19/month** |
