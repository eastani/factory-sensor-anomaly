variable "aws_region" {
  description = "AWS region for the state backend and IAM resources."
  type        = string
  default     = "ap-northeast-1"
}

variable "project_name" {
  description = "Tag value and resource-name prefix."
  type        = string
  default     = "factory-sensor-anomaly"
}

variable "github_repository" {
  description = "owner/repo string matched by the OIDC trust policy."
  type        = string
  default     = "eastani/factory-sensor-anomaly"
}

variable "allowed_branches" {
  description = "Git refs the deploy role may be assumed for."
  type        = list(string)
  default     = ["main"]
}

variable "state_bucket_name" {
  description = "Globally-unique S3 bucket name for Terraform state. Override per-account."
  type        = string
}

variable "lock_table_name" {
  description = "DynamoDB table for Terraform state locking."
  type        = string
  default     = "factory-sensor-anomaly-tfstate-locks"
}
