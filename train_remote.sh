#!/bin/bash

# Set error handling
set -e
set -o pipefail

echo "✅ Starting training setup on RunPod..."

# Create workspace directory
mkdir -p /workspace/mak
cd /workspace/mak

# Install rclone if not present
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

# Check if rclone is configured
if [ ! -f ~/.config/rclone/rclone.conf ]; then
    echo "⚠️ Rclone not configured. Please run 'rclone config' first."
    exit 1
fi

# Sync dataset & scripts from Wasabi to local pod
echo "Syncing files from Wasabi..."
rclone sync wasabi:mak-training /workspace/mak --progress

# Create and activate virtual environment
echo "Setting up Python environment..."
python -m venv venv
source venv/bin/activate

# Install requirements
echo "Installing Python packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate rclone

# Start training
echo "Starting training process..."
python train_local.py

# Sync results back to Wasabi
echo "Syncing results to Wasabi..."
rclone sync /workspace/mak/results wasabi:mak-training/results --progress
rclone sync /workspace/mak/logs wasabi:mak-training/logs --progress

echo "✅ Training complete! Results synced to Wasabi" 