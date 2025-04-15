# Set error action preference to stop on errors
$ErrorActionPreference = "Stop"

# Function to check if Python and required packages are installed
function Check-PythonEnvironment {
    try {
        python --version
        pip list | Select-String -Pattern "torch|transformers|datasets"
    }
    catch {
        Write-Host "Python not found. Please install Python 3.8 or later."
        exit 1
    }
}

# Function to install required packages
function Install-Requirements {
    Write-Host "Installing required packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets accelerate
}

# Function to check GPU availability
function Check-GPU {
    Write-Host "Checking GPU availability..."
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GPU check failed. Please ensure CUDA is properly installed."
        exit 1
    }
}

# Function to start training
function Start-Training {
    Write-Host "Starting training process..."
    $env:CUDA_VISIBLE_DEVICES = "0"
    python train_local.py
}

# Main execution
Write-Host "=== Starting Training Setup ==="
Check-PythonEnvironment
Install-Requirements
Check-GPU

# Start training
Write-Host "=== Starting Training Process ==="
Start-Training

Write-Host "Training complete!" 