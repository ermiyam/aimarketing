# GitHub Sync Script
Write-Host "Starting GitHub sync process..." -ForegroundColor Cyan

# Check if Git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

# Set variables
$GITHUB_URL = "https://github.com"
$PROJECT_NAME = "gen"
$BRANCH = "main"

# Check if remote exists
$remoteExists = git remote | Select-String "github"
if (!$remoteExists) {
    Write-Host "Adding GitHub remote..." -ForegroundColor Yellow
    git remote add github "$GITHUB_URL/$PROJECT_NAME.git"
}

# Pull latest changes
Write-Host "Pulling latest from GitHub..." -ForegroundColor Yellow
git pull github $BRANCH

# Add new changes
Write-Host "Adding new changes..." -ForegroundColor Yellow
git add .

# Commit changes
Write-Host "Committing changes..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Sync: $timestamp"

# Push to GitHub
Write-Host "Pushing to GitHub..." -ForegroundColor Yellow
git push github $BRANCH

Write-Host "Sync complete! Changes pushed to GitHub." -ForegroundColor Green 