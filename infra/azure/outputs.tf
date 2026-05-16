output "acr_login_server" {
  description = "ACR login server URL — also the image registry hostname."
  value       = azurerm_container_registry.main.login_server
}

output "api_url" {
  description = "HTTPS endpoint for the deployed API. curl <url>/healthz to verify."
  value       = "https://${azurerm_container_app.api.ingress[0].fqdn}"
}

output "resource_group" {
  description = "Resource group holding the application stack."
  value       = azurerm_resource_group.main.name
}

output "postgres_fqdn" {
  description = "Postgres Flexible Server FQDN (private)."
  value       = azurerm_postgresql_flexible_server.main.fqdn
  sensitive   = true
}
