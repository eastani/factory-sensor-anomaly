output "tenant_id" {
  description = "Set this in the GitHub Actions secret AZURE_TENANT_ID."
  value       = data.azurerm_subscription.current.tenant_id
}

output "subscription_id" {
  description = "Set this in the GitHub Actions secret AZURE_SUBSCRIPTION_ID."
  value       = var.subscription_id
}

output "client_id" {
  description = "Set this in the GitHub Actions secret AZURE_CLIENT_ID."
  value       = azuread_application.github.client_id
}

output "state_storage_account" {
  description = "Storage account holding tfstate. Used in the azurerm backend config."
  value       = azurerm_storage_account.tfstate.name
}

output "state_container" {
  description = "Container holding tfstate blobs."
  value       = azurerm_storage_container.tfstate.name
}

output "shared_resource_group" {
  description = "Resource Group for the bootstrap stack."
  value       = azurerm_resource_group.shared.name
}
