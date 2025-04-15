# Create .ssh directory if it doesn't exist
if (-not (Test-Path "$env:USERPROFILE\.ssh")) {
    New-Item -ItemType Directory -Path "$env:USERPROFILE\.ssh" -Force
}

# Generate new SSH key pair for RunPod
$keyPath = "$env:USERPROFILE\.ssh\runpod_key"
Write-Host "Generating new SSH key pair for RunPod..."
ssh-keygen -t rsa -b 4096 -f $keyPath -N '""'

# Display the public key
Write-Host "`nYour new public key is:"
Get-Content "$keyPath.pub"

# Create config file for RunPod
$configContent = @"
Host runpod
    HostName 213.173.105.9
    Port 22
    User root
    IdentityFile $keyPath
"@

$configContent | Out-File -FilePath "$env:USERPROFILE\.ssh\config" -Append -Encoding ASCII

Write-Host "`nSSH key pair generated successfully!"
Write-Host "Private key saved to: $keyPath"
Write-Host "Public key saved to: $keyPath.pub"
Write-Host "`nTo connect to RunPod, use: ssh runpod"
Write-Host "`nIMPORTANT: Copy the public key above and add it to your RunPod instance's authorized_keys file." 