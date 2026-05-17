################################################################################
# Application stack on Azure.
#
# Mirrors infra/aws/ at a similar level of detail. Differences from AWS:
#   - Container Apps scales to zero (true idle cost ~$0).
#   - Postgres Flexible Server is integrated to a delegated subnet via
#     private DNS — no public DB endpoint.
################################################################################

resource "azurerm_resource_group" "main" {
  name     = var.project_name
  location = var.location

  tags = {
    Project   = var.project_name
    ManagedBy = "terraform"
    Stack     = "app"
  }
}

# ----- Networking -------------------------------------------------------------

resource "azurerm_virtual_network" "main" {
  name                = "${var.project_name}-vnet"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  address_space       = ["10.20.0.0/16"]
}

# Container Apps' managed environment needs at least a /23 dedicated subnet.
resource "azurerm_subnet" "container_apps" {
  name                 = "container-apps"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.20.0.0/23"]
}

# Postgres Flexible Server requires a *delegated* subnet.
resource "azurerm_subnet" "postgres" {
  name                 = "postgres"
  resource_group_name  = azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = ["10.20.2.0/28"]

  delegation {
    name = "fs"
    service_delegation {
      name    = "Microsoft.DBforPostgreSQL/flexibleServers"
      actions = ["Microsoft.Network/virtualNetworks/subnets/join/action"]
    }
  }
}

resource "azurerm_private_dns_zone" "postgres" {
  name                = "${var.project_name}.postgres.database.azure.com"
  resource_group_name = azurerm_resource_group.main.name
}

resource "azurerm_private_dns_zone_virtual_network_link" "postgres" {
  name                  = "postgres-link"
  resource_group_name   = azurerm_resource_group.main.name
  private_dns_zone_name = azurerm_private_dns_zone.postgres.name
  virtual_network_id    = azurerm_virtual_network.main.id
}

# ----- ACR --------------------------------------------------------------------

resource "azurerm_container_registry" "main" {
  name                = replace("${var.project_name}acr", "-", "")
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = false
}

# ----- Postgres ---------------------------------------------------------------

resource "random_password" "db" {
  length           = 32
  special          = true
  override_special = "!*-_=+"
}

resource "azurerm_postgresql_flexible_server" "main" {
  name                          = "${var.project_name}-db"
  resource_group_name           = azurerm_resource_group.main.name
  location                      = azurerm_resource_group.main.location
  version                       = "16"
  administrator_login           = var.db_username
  administrator_password        = random_password.db.result
  sku_name                      = var.db_sku
  storage_mb                    = 32768
  storage_tier                  = "P4"
  delegated_subnet_id           = azurerm_subnet.postgres.id
  private_dns_zone_id           = azurerm_private_dns_zone.postgres.id
  public_network_access_enabled = false
  backup_retention_days         = 7
  zone                          = "1"

  depends_on = [azurerm_private_dns_zone_virtual_network_link.postgres]
}

resource "azurerm_postgresql_flexible_server_database" "app" {
  name      = var.db_name
  server_id = azurerm_postgresql_flexible_server.main.id
  charset   = "UTF8"
  collation = "en_US.utf8"
}
