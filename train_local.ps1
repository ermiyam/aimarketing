# Function to handle errors
function Write-ErrorAndExit {
    param([string]$Message)
    Write-Host "Error: $Message" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
try {
    . .\venv\Scripts\Activate.ps1
}
catch {
    Write-ErrorAndExit "Failed to activate virtual environment: $_"
}

# Set environment variables for better performance
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OMP_NUM_THREADS = "1"

# Start local training
Write-Host "Starting local training..."
try {
    python train_local.py
}
catch {
    Write-ErrorAndExit "Failed to start training: $_"
} 