variable "aws_region" {
  description = "AWS region for the application stack."
  type        = string
  default     = "ap-northeast-1"
}

variable "project_name" {
  description = "Tag value and resource-name prefix."
  type        = string
  default     = "factory-sensor-anomaly"
}

variable "image_tag" {
  description = "ECR image tag to deploy. CI sets this to the commit SHA."
  type        = string
  default     = "latest"
}

variable "db_username" {
  description = "RDS master username."
  type        = string
  default     = "anomaly"
}

variable "db_name" {
  description = "Application database name."
  type        = string
  default     = "anomaly"
}

variable "db_instance_class" {
  description = "RDS instance class. db.t4g.micro is the cheapest Postgres option."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage_gb" {
  description = "GB of GP3 storage allocated to RDS."
  type        = number
  default     = 20
}

variable "app_runner_cpu" {
  description = "App Runner instance CPU (in vCPU units, as a string per the API)."
  type        = string
  default     = "0.25 vCPU"
}

variable "app_runner_memory" {
  description = "App Runner instance memory."
  type        = string
  default     = "0.5 GB"
}

variable "alarm_email" {
  description = "Email to receive CloudWatch billing alerts. Leave empty to skip the alarm."
  type        = string
  default     = ""
}

variable "billing_alarm_thresholds_usd" {
  description = "USD thresholds at which CloudWatch billing alarms fire."
  type        = list(number)
  default     = [10, 30, 50]
}
