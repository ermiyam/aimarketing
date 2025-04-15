# Set error action preference
$ErrorActionPreference = "Stop"

# Function to check if rclone is installed
function Test-RcloneInstalled {
    try {
        $rcloneVersion = rclone version
        Write-Host "✅ Rclone is installed: $rcloneVersion"
        return $true
    }
    catch {
        return $false
    }
}

# Function to sync files to Wasabi
function Sync-ToWasabi {
    param (
        [string]$LocalPath = ".",
        [string]$RemotePath = "mak-training"
    )
    
    Write-Host "Syncing $LocalPath to wasabi:$RemotePath..."
    rclone sync $LocalPath "wasabi:$RemotePath" --progress
    Write-Host "✅ Sync complete"
}

# Function to sync files from Wasabi
function Sync-FromWasabi {
    param (
        [string]$RemotePath = "mak-training",
        [string]$LocalPath = "."
    )
    
    Write-Host "Syncing wasabi:$RemotePath to $LocalPath..."
    rclone sync "wasabi:$RemotePath" $LocalPath --progress
    Write-Host "✅ Sync complete"
}

# Main menu
function Show-Menu {
    Write-Host "`n=== Wasabi Sync Menu ==="
    Write-Host "1. Sync to Wasabi (upload)"
    Write-Host "2. Sync from Wasabi (download)"
    Write-Host "3. Exit"
    
    $choice = Read-Host "Enter your choice (1-3)"
    
    switch ($choice) {
        "1" {
            $localPath = Read-Host "Enter local path to sync (default: .)"
            if (-not $localPath) { $localPath = "." }
            $remotePath = Read-Host "Enter remote path (default: mak-training)"
            if (-not $remotePath) { $remotePath = "mak-training" }
            Sync-ToWasabi -LocalPath $localPath -RemotePath $remotePath
        }
        "2" {
            $remotePath = Read-Host "Enter remote path to sync (default: mak-training)"
            if (-not $remotePath) { $remotePath = "mak-training" }
            $localPath = Read-Host "Enter local path (default: .)"
            if (-not $localPath) { $localPath = "." }
            Sync-FromWasabi -RemotePath $remotePath -LocalPath $localPath
        }
        "3" {
            exit
        }
        default {
            Write-Host "Invalid choice. Please try again."
            Show-Menu
        }
    }
}

# Check if rclone is installed
try {
    $null = rclone version
}
catch {
    Write-Host "❌ Rclone is not installed or not in PATH."
    Write-Host "Please run the setup script first using:"
    Write-Host "  .\setup_wasabi.ps1"
    Write-Host ""
    Write-Host "If you've already run the setup script, try:"
    Write-Host "1. Close this PowerShell window"
    Write-Host "2. Open a new PowerShell window"
    Write-Host "3. Navigate to this directory:"
    Write-Host "   cd $PWD"
    Write-Host "4. Run the setup script again:"
    Write-Host "   .\setup_wasabi.ps1"
    exit 1
}

# Start menu
while ($true) {
    Show-Menu
} 