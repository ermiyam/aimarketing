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

# Function to install rclone
function Install-Rclone {
    Write-Host "Installing rclone..."
    
    # Download rclone
    $rcloneUrl = "https://downloads.rclone.org/rclone-current-windows-amd64.zip"
    $downloadPath = "$env:TEMP\rclone.zip"
    $extractPath = "$env:TEMP\rclone"
    
    # Download and extract
    Invoke-WebRequest -Uri $rcloneUrl -OutFile $downloadPath
    Expand-Archive -Path $downloadPath -DestinationPath $extractPath -Force
    
    # Find the rclone.exe in the extracted folder
    $rcloneExe = Get-ChildItem -Path $extractPath -Recurse -Filter "rclone.exe" | Select-Object -First 1
    
    if ($rcloneExe) {
        # Create a directory in user's home for rclone
        $userRcloneDir = "$env:USERPROFILE\rclone"
        if (-not (Test-Path $userRcloneDir)) {
            New-Item -ItemType Directory -Path $userRcloneDir -Force
        }
        
        # Copy rclone.exe to user's directory
        Copy-Item $rcloneExe.FullName -Destination $userRcloneDir -Force
        
        # Add to PATH
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if (-not $userPath.Contains($userRcloneDir)) {
            [Environment]::SetEnvironmentVariable("Path", "$userPath;$userRcloneDir", "User")
            Write-Host "✅ Rclone added to PATH"
            
            # Update PATH in current session
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        }
        
        # Cleanup
        Remove-Item $downloadPath -Force
        Remove-Item $extractPath -Recurse -Force
        Write-Host "✅ Rclone installed successfully"
        
        # Return the path to rclone.exe
        return "$userRcloneDir\rclone.exe"
    }
    else {
        throw "Failed to find rclone.exe in downloaded package"
    }
}

# Function to configure Wasabi
function Configure-Wasabi {
    param (
        [string]$RclonePath
    )
    
    Write-Host "Configuring Wasabi..."
    
    # Create rclone config directory if it doesn't exist
    $configDir = "$env:APPDATA\rclone"
    if (-not (Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force
    }
    
    # Prompt for Wasabi credentials
    $accessKey = Read-Host "Enter your Wasabi Access Key ID"
    $secretKey = Read-Host "Enter your Wasabi Secret Access Key" -AsSecureString
    
    # Convert secure string to plain text (temporarily)
    $BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secretKey)
    $plainSecret = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)
    
    # Create rclone config with correct region
    $config = @"
[wasabi]
type = s3
provider = Wasabi
access_key_id = $accessKey
secret_access_key = $plainSecret
region = us-east-1
endpoint = s3.wasabisys.com
"@
    
    # Save config
    $config | Out-File "$configDir\rclone.conf" -Encoding utf8
    
    # Clear sensitive data
    $plainSecret = $null
    [System.GC]::Collect()
    
    Write-Host "✅ Wasabi configuration saved"
}

# Function to test Wasabi connection
function Test-WasabiConnection {
    param (
        [string]$RclonePath
    )
    
    Write-Host "Testing Wasabi connection..."
    try {
        $result = & $RclonePath lsd wasabi: 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Successfully connected to Wasabi"
            return $true
        }
        else {
            Write-Host "❌ Failed to connect to Wasabi: $result"
            return $false
        }
    }
    catch {
        Write-Host "❌ Error testing Wasabi connection: $_"
        return $false
    }
}

# Function to create bucket if it doesn't exist
function Create-WasabiBucket {
    param (
        [string]$RclonePath,
        [string]$BucketName = "mak-training"
    )
    
    Write-Host "Creating bucket '$BucketName' if it doesn't exist..."
    try {
        $result = & $RclonePath mkdir "wasabi:$BucketName" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Bucket '$BucketName' is ready"
            return $true
        }
        else {
            # If bucket already exists, that's fine
            if ($result -like "*BucketAlreadyOwnedByYou*") {
                Write-Host "✅ Bucket '$BucketName' already exists"
                return $true
            }
            Write-Host "❌ Failed to create bucket: $result"
            return $false
        }
    }
    catch {
        Write-Host "❌ Error creating bucket: $_"
        return $false
    }
}

# Main execution
Write-Host "=== Wasabi Setup for Windows ==="

# Check and install rclone if needed
$rclonePath = $null
if (-not (Test-RcloneInstalled)) {
    $rclonePath = Install-Rclone
}
else {
    $rclonePath = (Get-Command rclone).Source
}

# Configure Wasabi
Configure-Wasabi -RclonePath $rclonePath

# Test connection and create bucket
if (Test-WasabiConnection -RclonePath $rclonePath) {
    Create-WasabiBucket -RclonePath $rclonePath
}

Write-Host "Setup complete! You can now use rclone to sync with Wasabi."
Write-Host "Example commands:"
Write-Host "  & '$rclonePath' sync ./mak wasabi:mak-training --progress"
Write-Host "  & '$rclonePath' sync wasabi:mak-training ./mak --progress"

# Create a PowerShell function for rclone
$rcloneFunction = @"
function rclone {
    param(
        [Parameter(Position=0, ValueFromRemainingArguments=$true)]
        [string[]]`$Arguments
    )
    & '$rclonePath' @Arguments
}
"@

# Add the function to the current session
Invoke-Expression $rcloneFunction

Write-Host "`nRclone is now available in this session. You can run commands like:"
Write-Host "rclone sync ./mak wasabi:mak-training --progress" 