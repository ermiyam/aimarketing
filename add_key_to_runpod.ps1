# RunPod instance details
$RUNPOD_IP = "213.173.105.9"
$RUNPOD_PORT = "41179"
$SSH_PORT = "22"
$RUNPOD_PASS = "0hick7ufhfjbjf1ioqn9"

# Get the public key
$publicKey = Get-Content "$env:USERPROFILE\.ssh\runpod_key.pub"

# Create a temporary script to run on RunPod
$tempScript = @"
#!/bin/bash
mkdir -p ~/.ssh
echo '$publicKey' >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
"@

# Save the temporary script
$tempScript | Out-File -FilePath "setup_key.sh" -Encoding ASCII

# Use plink to execute the script
$plinkPath = "C:\Program Files\PuTTY\plink.exe"
if (Test-Path $plinkPath) {
    & $plinkPath -ssh -P $SSH_PORT -pw $RUNPOD_PASS root@$RUNPOD_IP -m setup_key.sh
    Remove-Item setup_key.sh
    Write-Host "Public key added successfully to RunPod!"
    Write-Host "You can now connect using: ssh runpod"
} else {
    Write-Host "Please install PuTTY from https://www.putty.org/"
    exit 1
} 