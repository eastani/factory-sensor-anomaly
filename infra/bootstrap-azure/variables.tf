variable "subscription_id" {
  description = "Azure subscription ID. From `az account show --query id -o tsv`."
  type        = string
}

variable "location" {
  description = "Azure region for the state container and shared resources."
  type        = string
  default     = "japaneast"
}

variable "project_name" {
  description = "Tag value and resource-name prefix."
  type        = string
  default     = "factory-sensor-anomaly"
}

variable "github_repository" {
  description = "owner/repo string for the OIDC federated credential subject."
  type        = string
  default     = "eastani/factory-sensor-anomaly"
}

variable "allowed_branches" {
  description = "Git refs for which the deploy app registration may federate."
  type        = list(string)
  default     = ["main"]
}

variable "state_storage_account_name" {
  description = "Globally-unique Storage Account name for Terraform state (3-24 lowercase alphanumeric)."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9]{3,24}$", var.state_storage_account_name))
    error_message = "Storage Account names must be 3-24 lowercase alphanumeric characters."
  }
}

variable "state_container_name" {
  description = "Container inside the Storage Account holding tfstate blobs."
  type        = string
  default     = "tfstate"
}
