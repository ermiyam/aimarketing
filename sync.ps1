# GitLab Sync Script
Write-Host "Starting GitLab sync process..." -ForegroundColor Cyan

# Check if Git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is not installed. Please install it first." -ForegroundColor Red
    exit 1
}

# Set variables
$GITLAB_URL = "https://gitlab.com"
$PROJECT_NAME = "gen"
$BRANCH = "main"

# Check if remote exists
$remoteExists = git remote | Select-String "gitlab"
if (!$remoteExists) {
    Write-Host "Adding GitLab remote..." -ForegroundColor Yellow
    git remote add gitlab "$GITLAB_URL/$PROJECT_NAME.git"
}

# Pull latest changes
Write-Host "Pulling latest from GitLab..." -ForegroundColor Yellow
git pull gitlab $BRANCH

# Add new changes
Write-Host "Adding new changes..." -ForegroundColor Yellow
git add .

# Commit changes
Write-Host "Committing changes..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Sync: $timestamp"

# Push to GitLab
Write-Host "Pushing to GitLab..." -ForegroundColor Yellow
git push gitlab $BRANCH

Write-Host "Sync complete! Changes pushed to GitLab." -ForegroundColor Green 