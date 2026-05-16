output "ecr_repository_url" {
  description = "ECR image URI to push to."
  value       = data.aws_ecr_repository.api.repository_url
}

output "api_service_url" {
  description = "HTTPS endpoint for the deployed API. curl <url>/healthz to verify."
  value       = "https://${aws_apprunner_service.api.service_url}"
}

output "api_service_arn" {
  description = "App Runner service ARN — useful for CI start-deployment calls."
  value       = aws_apprunner_service.api.arn
}

output "rds_endpoint" {
  description = "Postgres endpoint (private)."
  value       = aws_db_instance.main.address
  sensitive   = true
}

output "db_secret_arn" {
  description = "Secrets Manager ARN holding the Postgres credentials."
  value       = aws_secretsmanager_secret.db_url.arn
}
