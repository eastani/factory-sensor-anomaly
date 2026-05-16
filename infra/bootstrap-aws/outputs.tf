output "tfstate_bucket" {
  description = "S3 bucket holding Terraform remote state."
  value       = aws_s3_bucket.tfstate.bucket
}

output "tflock_table" {
  description = "DynamoDB table used for Terraform state locking."
  value       = aws_dynamodb_table.tflocks.name
}

output "github_deploy_role_arn" {
  description = "Paste this into the GitHub Actions secret AWS_DEPLOY_ROLE_ARN."
  value       = aws_iam_role.github_deploy.arn
}

output "aws_region" {
  description = "Region for the application stack to reuse."
  value       = var.aws_region
}

output "ecr_repository_url" {
  description = "ECR image URI for CI to push to."
  value       = aws_ecr_repository.api.repository_url
}
