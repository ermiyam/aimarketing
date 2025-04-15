# RunPod instance details
$RUNPOD_IP = "213.173.105.9"
$RUNPOD_PORT = "41179"
$SSH_PORT = "22"
$RUNPOD_PASS = "0hick7ufhfjbjf1ioqn9"

# SSH key details
$SSH_PUB_KEY = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDhEcFs+T6JelaMQyY0Ps233sfnPXHMz8PoqQYbak5KTHbVa9v/6K8j9Lrcd0DR25i8prqYYu8fXBabuO2StQz5/HhLovUmFwZUmGsVAtXXPAnIsAARAtuRpfD9YYoOBmvFtwkHwi57HT6vGObWKHF0rfRzOV4UOFMLRvEqjI2ozhgodDTbHjr9/hOFfRhgIauCQ2WW1h+pESzu+Z8dEVXiBipVGEz8NssRicVfGSxigi30lZKK7XW5ai7apAQQCem6nFq/i95ioAIq2BPa1OSRWlh+F2DKIR3yrP/HcB5H7XQTL83ASbcSFP1WUaHw0L0WT4R7NKrk2yBHJ5KjdAgT0VIUmsrpzTlzYxZH4tCQt5WWuP/oMHyeG1w0O3O6271RYK37Sl6vqcJWYVpZAnNpRY4wKPE7Gt2SPSaUKApUH0uPfYXQz/GOdzXBVAXXZVBNjJ+VK97ky2MPWB8n8zzdM1pC/cBiVtzRGlE7e6Yx5tRSM6tDTYriHREp5qwsraTBoWLJVFuJg8yjrv3wRLcCk7FI+rkt7VfdYUAZvQevqFzt4wQpHEzwDGV7gPPFhT++xaHehK68K2gD4+LZ7NCu9Gm/9TOfO/TILCV7qO/PoGV8UdliBuc9rD0JwDZFCBfQoiCq2xt7RdYvW2ECChFItez6opzEb3VBItV+Q5D7Jw== ermiyamousavian@gmail.com"

# Check if SSH is installed
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Host "SSH is not installed. Please install OpenSSH Client from Windows Features."
    exit 1
}

# Create .ssh directory if it doesn't exist
if (-not (Test-Path "$env:USERPROFILE\.ssh")) {
    New-Item -ItemType Directory -Path "$env:USERPROFILE\.ssh" -Force
}

# Save the public key
$SSH_PUB_KEY | Out-File -FilePath "$env:USERPROFILE\.ssh\id_rsa.pub" -Encoding ASCII

# Function to execute SSH command with password
function Invoke-SSHWithPassword {
    param (
        [string]$Command,
        [string]$Password
    )
    
    $plinkPath = "C:\Program Files\PuTTY\plink.exe"
    if (-not (Test-Path $plinkPath)) {
        Write-Host "Please install PuTTY from https://www.putty.org/"
        exit 1
    }
    
    & $plinkPath -ssh -P $SSH_PORT -pw $Password root@$RUNPOD_IP $Command
}

# Copy public key to RunPod
Write-Host "Setting up SSH key authentication..."
Invoke-SSHWithPassword -Command "mkdir -p ~/.ssh && echo '$SSH_PUB_KEY' >> ~/.ssh/authorized_keys" -Password $RUNPOD_PASS

# Upload files using SCP
Write-Host "Uploading files to RunPod..."
$scpPath = "C:\Program Files\Git\usr\bin\scp.exe"
if (Test-Path $scpPath) {
    & $scpPath -P $SSH_PORT -i "$env:USERPROFILE\.ssh\id_rsa" -r ./* root@${RUNPOD_IP}:/workspace/
} else {
    Write-Host "Please install Git for Windows from https://git-scm.com/download/win"
    exit 1
}

# SSH into RunPod and run setup
Write-Host "Connecting to RunPod and running setup..."
Invoke-SSHWithPassword -Command "cd /workspace && chmod +x runpod_setup.sh && ./runpod_setup.sh" -Password $RUNPOD_PASS

Write-Host "Setup complete! You can now access your RunPod instance at:"
Write-Host "Web Interface: http://$RUNPOD_IP`:$RUNPOD_PORT"
Write-Host "SSH: ssh -p $SSH_PORT -i `"$env:USERPROFILE\.ssh\id_rsa`" root@$RUNPOD_IP" 