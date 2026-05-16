################################################################################
# bootstrap-azure — one-time setup paralleling infra/bootstrap-aws.
#
# Creates:
#   - Resource Group for shared / state resources
#   - Storage Account + Container for Terraform remote state
#   - App Registration + Service Principal that GitHub Actions federates to
#   - Federated credential restricting OIDC to specific branches
#   - Contributor role assignment scoped to the subscription
################################################################################

data "azurerm_subscription" "current" {}

resource "azurerm_resource_group" "shared" {
  name     = "${var.project_name}-shared"
  location = var.location

  tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
    Stack     = "bootstrap"
  }
}

# ----- Terraform remote state backend -----------------------------------------

resource "azurerm_storage_account" "tfstate" {
  name                            = var.state_storage_account_name
  resource_group_name             = azurerm_resource_group.shared.name
  location                        = azurerm_resource_group.shared.location
  account_tier                    = "Standard"
  account_replication_type        = "LRS"
  account_kind                    = "StorageV2"
  https_traffic_only_enabled      = true
  min_tls_version                 = "TLS1_2"
  shared_access_key_enabled       = true
  allow_nested_items_to_be_public = false
  blob_properties {
    versioning_enabled = true
    delete_retention_policy {
      days = 7
    }
  }

  tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
    Stack     = "bootstrap"
  }
}

resource "azurerm_storage_container" "tfstate" {
  name                  = var.state_container_name
  storage_account_id    = azurerm_storage_account.tfstate.id
  container_access_type = "private"
}

# ----- App Registration for GitHub Actions OIDC -------------------------------

resource "azuread_application" "github" {
  display_name = "${var.project_name}-github-deploy"
  description  = "Federated identity for GitHub Actions to deploy ${var.project_name}."
}

resource "azuread_service_principal" "github" {
  client_id = azuread_application.github.client_id
}

resource "azuread_application_federated_identity_credential" "github" {
  for_each       = toset(var.allowed_branches)
  application_id = azuread_application.github.id
  display_name   = "github-${each.value}"
  description    = "Federation for refs/heads/${each.value} on ${var.github_repository}."
  audiences      = ["api://AzureADTokenExchange"]
  issuer         = "https://token.actions.githubusercontent.com"
  subject        = "repo:${var.github_repository}:ref:refs/heads/${each.value}"
}

# Contributor at subscription scope is broader than ideal but is the
# de-facto minimum for "apply Terraform that creates resource groups + RBAC".
# Tighten by scoping per-resource-group once Phase 3 lands.
resource "azurerm_role_assignment" "github_contributor" {
  scope                = data.azurerm_subscription.current.id
  role_definition_name = "Contributor"
  principal_id         = azuread_service_principal.github.object_id
}

# Storage Blob Data Contributor on the tfstate account is the role that lets
# OIDC-authenticated terraform read/write state blobs (azurerm backend with
# use_azuread_auth=true).
resource "azurerm_role_assignment" "github_state_blob" {
  scope                = azurerm_storage_account.tfstate.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azuread_service_principal.github.object_id
}
