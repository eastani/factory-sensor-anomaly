variable "subscription_id" {
  description = "Azure subscription ID."
  type        = string
}

variable "location" {
  description = "Azure region for application resources."
  type        = string
  default     = "japaneast"
}

variable "project_name" {
  description = "Tag value and resource-name prefix."
  type        = string
  default     = "factory-sensor-anomaly"
}

variable "image_tag" {
  description = "ACR image tag to deploy. CI sets this to the commit SHA."
  type        = string
  default     = "latest"
}

variable "db_username" {
  description = "Postgres administrator login."
  type        = string
  default     = "anomaly"
}

variable "db_name" {
  description = "Application database name."
  type        = string
  default     = "anomaly"
}

variable "db_sku" {
  description = "Azure DB for PostgreSQL Flexible Server SKU. Burstable B1ms is cheapest."
  type        = string
  default     = "B_Standard_B1ms"
}

variable "container_app_cpu" {
  description = "Container Apps vCPU per replica."
  type        = number
  default     = 0.25
}

variable "container_app_memory" {
  description = "Container Apps memory per replica."
  type        = string
  default     = "0.5Gi"
}

variable "container_app_min_replicas" {
  description = "Minimum replicas — 0 enables true scale-to-zero (~$0 idle)."
  type        = number
  default     = 0
}

variable "container_app_max_replicas" {
  description = "Maximum replicas. Cap to keep accidental traffic from running up the bill."
  type        = number
  default     = 2
}
