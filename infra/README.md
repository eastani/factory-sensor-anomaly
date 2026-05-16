# Infrastructure as Code

Two parallel Terraform stacks, AWS and Azure, mirroring the decision in
[ADR-0005](../docs/adr/0005-cloud-architecture.md). They deploy **only the
API service** — the dashboard stays local. Both stacks are designed to be
torn down with one `terraform destroy` so this stays a portfolio project,
not a recurring bill.

```
infra/
├── README.md           ← you are here
├── bootstrap-aws/      ← one-time: state bucket, lock table, IAM admin
├── aws/                ← the application stack: ECR + App Runner + RDS
└── azure/              ← Azure parallel stack: ACR + Container Apps + DB
```

## ⚠️ AWS root user warning

If your only AWS credential right now is the **root user**:

1. **Add MFA to root immediately** (Console → IAM → My security credentials).
2. **Never create an access key for root.** Delete any that exist.
3. Use root **once** to create an IAM user (or, preferred, an IAM Identity
   Center user) with the `AdministratorAccess` policy, MFA-enforced, and a
   programmatic access key (or, even better, IAM Identity Center SSO).
4. Configure `aws configure` (or `aws configure sso`) with that user.
5. Log out of root. Do not log back in unless you are rotating keys or
   closing the account.

Root account compromise is unrecoverable. Do not skip this.

## One-time AWS bootstrap

Before the application stack can run, it needs:

- An S3 bucket for Terraform state.
- A DynamoDB table for state locking.
- An IAM OIDC identity provider for GitHub Actions.
- An IAM role the GitHub workflow can assume.

Run from a machine logged in as your admin IAM user:

```bash
cd infra/bootstrap-aws
terraform init
terraform apply
```

Outputs include the role ARN to paste into the GitHub Actions secret
`AWS_DEPLOY_ROLE_ARN`. **This bootstrap is the only step that uses
locally-stored AWS credentials. Everything afterwards runs through OIDC.**

## Deploying the application (AWS)

After bootstrap, deploys happen via GitHub Actions on push to `main`
(`.github/workflows/deploy-aws.yml`). To deploy manually:

```bash
cd infra/aws
terraform init
terraform apply       # creates ECR + App Runner + RDS
```

App Runner returns an HTTPS URL like
`https://<id>.<region>.awsapprunner.com`. Smoke-test:

```bash
curl https://<id>.<region>.awsapprunner.com/healthz
```

## Tear-down

```bash
cd infra/aws
terraform destroy
```

The bootstrap stack stays — destroying it would erase Terraform state for
everyone. Keep it unless you are decommissioning the project entirely.

## Cost monitoring

Every resource is tagged `Project=factory-sensor-anomaly` and
`ManagedBy=terraform`. A CloudWatch billing alarm fires at $10 / $30 / $50
thresholds; recipient is set via the `alarm_email` variable.

## Azure parallel stack

See [`azure/README.md`](azure/README.md) — same shape, different services.
