# RunPod instance details
$RUNPOD_IP = "213.173.105.9"
$RUNPOD_PORT = "41179"
$SSH_PORT = "22"
$RUNPOD_PASS = "0hick7ufhfjbjf1ioqn9"

# Your existing public key
$publicKey = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC76Q8UVoz/2YiCUL6MVU4RhWyEgegnbaOIsKri25aZsDbERxnltoa7x6Xs/zDBnpVH1QfKt1ZWzpssRXDtKdfwGRomA6gi9X2bSaQjLGHPPW/1BTKlKzL24TxYstonTh+n/NHuXwQFl2IcxiL0pUjNc6BY3NIWDMlwWhPcIbsH+vt4UDFBEa7PQ90iPmSHKqZOPQzVBDNtFSpgSF6kjDRHv2ZobqeknDF2FrairBqoTkbKEMrI9+UVFkRNbDMLDrJDTwk3/THem/UBWDNQa0rtfRH0QMUlrc8caCxq+utZvh/jo7fCnF5Kbr/N9OcX1yRsX/+wQvRWpVytNPBDbrLMMHq6GhNPLLtgIVR7qdmu3Y8PDyF+fgSfDXM9VnxV/qLaY7mRZfPk9qF3oE6IRR+qDJqlfJUEBd5RVR0aBn7MLO3KogSE1xc5pXkHTRTtIzExmJqAT5Lf4OqY3j6l8CxoJY9qczEbOhzRoWFGHiB8DR9FFvL/BnpKyF2kvE9DR/KlasnznuP0yTEjMvKuHpOzOB7xbereUYdsrheO3b8BOIbxek7O5KuaRBPp+Yf2vnyQ+ulra87rgJxmJ0rAWNwbK19D7qldhng1A8LZFGVy5xOKVZcHOEraFC1RgBhP9Gb3yrUyR/Kw71wop0I/Hc8ZruD8dNG/ymirP9Js2rea4Q== ermiy@DESKTOP-R0M4PCH"

# Create .ssh directory if it doesn't exist
if (-not (Test-Path "$env:USERPROFILE\.ssh")) {
    New-Item -ItemType Directory -Path "$env:USERPROFILE\.ssh" -Force
}

# Save the public key
$publicKey | Out-File -FilePath "$env:USERPROFILE\.ssh\runpod_key.pub" -Encoding ASCII

# Create SSH config for RunPod
$configContent = @"
Host runpod
    HostName $RUNPOD_IP
    Port $SSH_PORT
    User root
    IdentityFile $env:USERPROFILE\.ssh\id_rsa
"@

$configContent | Out-File -FilePath "$env:USERPROFILE\.ssh\config" -Append -Encoding ASCII

# Create script to add key to RunPod
$tempScript = @"
#!/bin/bash
mkdir -p ~/.ssh
echo '$publicKey' >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
"@

# Save and execute the setup script
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