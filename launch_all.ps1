# Launch script for Mak AI system

# Load configuration
. .\config.ps1

# Create necessary directories
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "models"
New-Item -ItemType Directory -Force -Path "data"

# Start components
Write-Host "Starting Mak AI system components..."

# Start orchestrator
Start-Process python -ArgumentList "src/mak_orchestrator.py" -NoNewWindow

# Start training
Start-Process python -ArgumentList "src/train_mak.py" -NoNewWindow

# Start self-optimization
Start-Process python -ArgumentList "src/self_optimize_learning.py" -NoNewWindow

# Start scraper trainer
Start-Process python -ArgumentList "src/scraper_trainer.py" -NoNewWindow

# Start LLM inference
Start-Process python -ArgumentList "src/llm_inference.py" -NoNewWindow

# Start Wasabi sync
Start-Process python -ArgumentList "src/wasabi_sync.py" -NoNewWindow

# Start Telegram bot
Start-Process python -ArgumentList "src/telegram_bot.py" -NoNewWindow

# Start dashboard
Start-Process python -ArgumentList "src/dashboard.py" -NoNewWindow

Write-Host "All components started successfully"
Write-Host "Check logs directory for component outputs"
