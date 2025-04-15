# Azure Deployment Script
Write-Host "Starting Azure deployment process..." -ForegroundColor Cyan

# Check if Azure CLI is installed
if (!(Get-Command az -ErrorAction SilentlyContinue)) {
    Write-Host "Azure CLI is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

# Login to Azure
Write-Host "Logging into Azure..." -ForegroundColor Yellow
az login

# Set variables
$resourceGroup = "gen-rg"
$location = "eastus"
$appName = "gen-app"
$containerRegistry = "genregistry"
$keyVaultName = "gen-kv"
$logAnalyticsWorkspace = "gen-logs"

# Create resource group
Write-Host "Creating resource group..." -ForegroundColor Yellow
az group create --name $resourceGroup --location $location

# Create Azure Container Registry
Write-Host "Creating container registry..." -ForegroundColor Yellow
az acr create --resource-group $resourceGroup --name $containerRegistry --sku Basic --admin-enabled true

# Create Key Vault
Write-Host "Creating Key Vault..." -ForegroundColor Yellow
az keyvault create --name $keyVaultName --resource-group $resourceGroup --location $location

# Create Log Analytics Workspace
Write-Host "Creating Log Analytics workspace..." -ForegroundColor Yellow
az monitor log-analytics workspace create --resource-group $resourceGroup --workspace-name $logAnalyticsWorkspace --location $location

# Build and push Docker image
Write-Host "Building and pushing Docker image..." -ForegroundColor Yellow
az acr build --registry $containerRegistry --image $appName:latest .

# Create App Service plan
Write-Host "Creating App Service plan..." -ForegroundColor Yellow
az appservice plan create --name "$appName-plan" --resource-group $resourceGroup --sku B1 --is-linux

# Create Web App
Write-Host "Creating Web App..." -ForegroundColor Yellow
az webapp create --resource-group $resourceGroup --plan "$appName-plan" --name $appName --deployment-container-image-name "$containerRegistry.azurecr.io/$appName:latest"

# Configure environment variables
Write-Host "Configuring environment variables..." -ForegroundColor Yellow
az webapp config appsettings set --name $appName --resource-group $resourceGroup --settings `
    AZURE_KEY_VAULT_URL="https://$keyVaultName.vault.azure.net/" `
    AZURE_WORKSPACE_ID="$(az monitor log-analytics workspace show --resource-group $resourceGroup --workspace-name $logAnalyticsWorkspace --query customerId -o tsv)" `
    AZURE_SUBSCRIPTION_ID="$(az account show --query id -o tsv)"

# Enable logging
Write-Host "Enabling logging..." -ForegroundColor Yellow
az webapp log config --name $appName --resource-group $resourceGroup --application-logging filesystem

Write-Host "Deployment complete! Your app is available at: https://$appName.azurewebsites.net" -ForegroundColor Green 