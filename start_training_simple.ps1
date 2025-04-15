# RunPod connection details
$RUNPOD_IP = "213.173.105.9"  # The actual IP address
$SSH_PORT = "27179"           # The SSH port number
$MAX_RETRIES = 3
$RETRY_DELAY_SECONDS = 5

# Function to handle errors
function Write-ErrorAndExit {
    param([string]$Message)
    Write-Host "Error: $Message" -ForegroundColor Red
    exit 1
}

# Function to test SSH connection
function Test-SSHConnection {
    param (
        [string]$IP,
        [string]$Port
    )
    
    Write-Host "Testing SSH connection to ${IP}:${Port}..."
    try {
        $result = ssh -p $Port -o "ConnectTimeout=5" -o "BatchMode=yes" "root@${IP}" "echo 'Connection successful'" 2>&1
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
try {
    . .\venv\Scripts\Activate.ps1
}
catch {
    Write-ErrorAndExit "Failed to activate virtual environment: $_"
}

# Check if we should run distributed training
$USE_DISTRIBUTED = $true  # Set to $false to run only on local GPU

if ($USE_DISTRIBUTED) {
    Write-Host "Starting distributed training setup..."
    
    # Set environment variables for distributed training
    $env:MASTER_ADDR = $RUNPOD_IP
    $env:MASTER_PORT = "29500"
    $env:WORLD_SIZE = "2"
    $env:CUDA_VISIBLE_DEVICES = "0"
    $env:OMP_NUM_THREADS = "1"

    # Verify SSH connection with retries
    Write-Host "Verifying RunPod connection..."
    $connected = $false
    for ($i = 1; $i -le $MAX_RETRIES; $i++) {
        try {
            $result = ssh -p $SSH_PORT -o "ConnectTimeout=5" -o "BatchMode=yes" "root@${RUNPOD_IP}" "echo 'Connection successful'" 2>&1
            if ($LASTEXITCODE -eq 0) {
                $connected = $true
                break
            }
        }
        catch {
            if ($i -lt $MAX_RETRIES) {
                Write-Host "Connection attempt $i failed. Retrying in $RETRY_DELAY_SECONDS seconds..."
                Start-Sleep -Seconds $RETRY_DELAY_SECONDS
            }
        }
    }

    if (-not $connected) {
        Write-ErrorAndExit @"
Failed to connect to RunPod after $MAX_RETRIES attempts.
Please verify:
1. Your RunPod instance is running
2. The IP address ($RUNPOD_IP) and port ($SSH_PORT) are correct
3. Your SSH key is properly set up
4. The RunPod instance's SSH service is running

You can test the connection manually with:
ssh -p $SSH_PORT root@$RUNPOD_IP
"@
    }

    # Upload training files to RunPod
    Write-Host "Uploading training files to RunPod..."
    try {
        # Upload config file first
        scp -P $SSH_PORT config.yaml "root@${RUNPOD_IP}:/workspace/"
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload config.yaml" }
        
        # Upload training script
        scp -P $SSH_PORT train_distributed.py "root@${RUNPOD_IP}:/workspace/"
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload train_distributed.py" }
        
        # Create and upload the RunPod training script with environment variables
        @"
#!/bin/bash
export MASTER_ADDR=$RUNPOD_IP
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
cd /workspace
python train_distributed.py
"@ | Out-File -FilePath "runpod_train.sh" -Encoding ASCII
        
        scp -P $SSH_PORT runpod_train.sh "root@${RUNPOD_IP}:/workspace/"
        if ($LASTEXITCODE -ne 0) { throw "Failed to upload runpod_train.sh" }
        
        # Make the script executable
        ssh -p $SSH_PORT "root@${RUNPOD_IP}" "chmod +x /workspace/runpod_train.sh"
        if ($LASTEXITCODE -ne 0) { throw "Failed to make script executable" }
    }
    catch {
        Write-ErrorAndExit "Failed to upload files: $_"
    }

    # Start RunPod training in background
    Write-Host "Starting RunPod training node..."
    try {
        $runpodJob = Start-Job -ScriptBlock {
            param($SSH_PORT, $RUNPOD_IP)
            ssh -p $SSH_PORT "root@${RUNPOD_IP}" "/workspace/runpod_train.sh"
        } -ArgumentList $SSH_PORT, $RUNPOD_IP
    }
    catch {
        Write-ErrorAndExit "Failed to start RunPod training: $_"
    }

    # Start local training
    Write-Host "Starting local training node..."
    try {
        $env:RANK = "0"
        python train_distributed.py
    }
    catch {
        Write-ErrorAndExit "Failed to start local training: $_"
    }

    # Cleanup
    Write-Host "Training complete. Cleaning up..."
    Stop-Job -Job $runpodJob
    Remove-Job -Job $runpodJob
    Remove-Item -Path "runpod_train.sh" -ErrorAction SilentlyContinue
}
else {
    # Single GPU training
    Write-Host "Starting single GPU training..."
    try {
        python train_distributed.py
    }
    catch {
        Write-ErrorAndExit "Failed to start training: $_"
    }
} 