# RunPod instance details
$RUNPOD_IP = "213.173.105.9"
$RUNPOD_PORT = "41179"
$SSH_PORT = "22"
$RUNPOD_PASS = "0hick7ufhfjbjf1ioqn9"

# Check if PuTTY is installed
$puttyPath = "C:\Program Files\PuTTY\putty.exe"
if (-not (Test-Path $puttyPath)) {
    Write-Host "Please install PuTTY from https://www.putty.org/"
    exit 1
}

# Create session configuration
$sessionName = "RunPod_Mak_Research"
$puttyConfig = @"
HostName=$RUNPOD_IP
Port=$SSH_PORT
Protocol=ssh
UserName=root
Password=$RUNPOD_PASS
"@

# Save PuTTY session configuration
$puttyConfig | Out-File -FilePath "$env:APPDATA\PuTTY\Sessions\$sessionName" -Encoding ASCII

Write-Host "Connecting to RunPod instance..."
& $puttyPath -load "$sessionName" 