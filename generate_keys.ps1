# Check if PuTTYgen is installed
$puttygenPath = "C:\Program Files\PuTTY\puttygen.exe"
if (-not (Test-Path $puttygenPath)) {
    Write-Host "Please install PuTTY from https://www.putty.org/"
    exit 1
}

# Create .ssh directory if it doesn't exist
if (-not (Test-Path "$env:USERPROFILE\.ssh")) {
    New-Item -ItemType Directory -Path "$env:USERPROFILE\.ssh" -Force
}

# Generate new SSH key pair
Write-Host "Generating new SSH key pair..."
& $puttygenPath -t rsa -b 4096 -o "$env:USERPROFILE\.ssh\id_rsa.ppk" -C "ermiyamousavian@gmail.com"

# Convert PPK to OpenSSH format
& $puttygenPath "$env:USERPROFILE\.ssh\id_rsa.ppk" -O private-openssh -o "$env:USERPROFILE\.ssh\id_rsa"
& $puttygenPath "$env:USERPROFILE\.ssh\id_rsa.ppk" -O public-openssh -o "$env:USERPROFILE\.ssh\id_rsa.pub"

# Display the public key
Write-Host "`nYour new public key is:"
Get-Content "$env:USERPROFILE\.ssh\id_rsa.pub"

Write-Host "`nKeys have been generated and saved to:"
Write-Host "Private key (PPK): $env:USERPROFILE\.ssh\id_rsa.ppk"
Write-Host "Private key (OpenSSH): $env:USERPROFILE\.ssh\id_rsa"
Write-Host "Public key: $env:USERPROFILE\.ssh\id_rsa.pub"

Write-Host "`nPlease save these keys in a secure location!" 