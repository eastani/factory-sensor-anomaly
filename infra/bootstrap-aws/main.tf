################################################################################
# bootstrap-aws — one-time setup that the application stack depends on.
#
# Creates:
#   - S3 bucket for Terraform remote state (versioned, encrypted, block-public)
#   - DynamoDB table for state locking
#   - GitHub Actions OIDC provider (one per AWS account; idempotent if exists)
#   - IAM role the GitHub workflow assumes via OIDC
#
# Re-run safely: every resource is idempotent under terraform apply.
################################################################################

data "aws_caller_identity" "current" {}

# ----- Terraform state backend ------------------------------------------------

resource "aws_s3_bucket" "tfstate" {
  bucket        = var.state_bucket_name
  force_destroy = false
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tflocks" {
  name         = var.lock_table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

# ----- GitHub Actions OIDC ----------------------------------------------------
#
# AWS deprecated the thumbprint requirement in late 2023; an empty list is the
# documented modern form.

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

# ----- IAM role assumable from GitHub Actions ---------------------------------

data "aws_iam_policy_document" "github_assume" {
  statement {
    effect = "Allow"

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    actions = ["sts:AssumeRoleWithWebIdentity"]

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Lock the role to only the listed refs. No PRs from forks; no other branches.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        for ref in var.allowed_branches :
        "repo:${var.github_repository}:ref:refs/heads/${ref}"
      ]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name                 = "${var.project_name}-github-deploy"
  assume_role_policy   = data.aws_iam_policy_document.github_assume.json
  description          = "Assumed by GitHub Actions to deploy ${var.project_name}."
  max_session_duration = 3600
}

# Permissions the deploy role needs:
#   - ECR push (image build/push)
#   - App Runner update (deploy)
#   - S3 + DynamoDB on the state backend
#   - RDS describe (so terraform plan can read outputs)
data "aws_iam_policy_document" "deploy" {
  statement {
    sid = "EcrPushPull"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart",
      "ecr:CompleteLayerUpload",
      "ecr:PutImage",
      "ecr:DescribeRepositories",
      "ecr:DescribeImages",
      "ecr:CreateRepository",
    ]
    resources = ["*"]
  }

  statement {
    sid = "AppRunnerDeploy"
    actions = [
      "apprunner:CreateService",
      "apprunner:DescribeService",
      "apprunner:ListServices",
      "apprunner:StartDeployment",
      "apprunner:UpdateService",
      "apprunner:DeleteService",
      "apprunner:TagResource",
      "apprunner:UntagResource",
      "apprunner:ListTagsForResource",
      "apprunner:CreateVpcConnector",
      "apprunner:DescribeVpcConnector",
      "apprunner:DeleteVpcConnector",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "IamPassToAppRunner"
    actions   = ["iam:PassRole"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["build.apprunner.amazonaws.com", "tasks.apprunner.amazonaws.com"]
    }
  }

  statement {
    sid       = "RdsReadOnly"
    actions   = ["rds:Describe*", "rds:ListTagsForResource"]
    resources = ["*"]
  }

  statement {
    sid       = "TfState"
    actions   = ["s3:ListBucket", "s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [aws_s3_bucket.tfstate.arn, "${aws_s3_bucket.tfstate.arn}/*"]
  }

  statement {
    sid       = "TfLocks"
    actions   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem", "dynamodb:DescribeTable"]
    resources = [aws_dynamodb_table.tflocks.arn]
  }
}

resource "aws_iam_role_policy" "deploy" {
  name   = "${var.project_name}-deploy"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.deploy.json
}

# ----- ECR --------------------------------------------------------------------
# Lives in bootstrap (not the app stack) because chicken-and-egg: CI must be
# able to push an image to ECR before the app stack's terraform apply runs
# (App Runner pulls *during* apply, so the image must already exist).

resource "aws_ecr_repository" "api" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the most recent 10 images."
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = { type = "expire" }
      }
    ]
  })
}
