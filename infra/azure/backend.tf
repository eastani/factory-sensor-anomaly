# Remote state backend in the bootstrap Storage Account.
# Initialise with: terraform init -backend-config=backend.hcl

terraform {
  backend "azurerm" {
    key              = "factory-sensor-anomaly/azure/terraform.tfstate"
    use_azuread_auth = true
    # resource_group_name, storage_account_name, container_name are
    # supplied via backend.hcl so account-specific names are not committed.
  }
}
