# Multi-Platform Sync Script
Write-Host "Starting multi-platform sync process..." -ForegroundColor Cyan

# Check if Git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

# Set variables
$GITLAB_URL = "https://gitlab.com"
$GITHUB_URL = "https://github.com"
$PROJECT_NAME = "gen"
$BRANCH = "main"

# Function to setup remote
function Setup-Remote {
    param (
        [string]$platform,
        [string]$url
    )
    
    $remoteExists = git remote | Select-String $platform
    if (!$remoteExists) {
        Write-Host "Adding $platform remote..." -ForegroundColor Yellow
        git remote add $platform "$url/$PROJECT_NAME.git"
    }
}

# Function to sync with platform
function Sync-Platform {
    param (
        [string]$platform
    )
    
    Write-Host "Syncing with $platform..." -ForegroundColor Yellow
    git pull $platform $BRANCH
    git push $platform $BRANCH
}

# Setup remotes
Setup-Remote -platform "gitlab" -url $GITLAB_URL
Setup-Remote -platform "github" -url $GITHUB_URL

# Add new changes
Write-Host "Adding new changes..." -ForegroundColor Yellow
git add .

# Commit changes
Write-Host "Committing changes..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Sync: $timestamp"

# Sync with both platforms
Sync-Platform -platform "gitlab"
Sync-Platform -platform "github"

Write-Host "Sync complete! Changes pushed to all platforms." -ForegroundColor Green 