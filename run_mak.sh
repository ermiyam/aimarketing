#!/bin/bash

# === run_mak.sh ===
# Full command center launcher for MAK (Ermiya's AI platform)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to log messages with timestamp
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a process is running
check_process() {
    if pgrep -f "$1" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to start process with logging
start_process() {
    local name=$1
    local cmd=$2
    local log_file=$3
    
    log "${BLUE}Starting $name...${NC}"
    nohup $cmd &> "$log_file" &
    sleep 2
    
    if check_process "$cmd"; then
        log "${GREEN}$name started successfully${NC}"
    else
        log "${RED}Failed to start $name${NC}"
        exit 1
    fi
}

# Create necessary directories
mkdir -p logs
mkdir -p data/feedback
mkdir -p data/knowledge

# Step 1: Activate virtual environment (if exists)
if [ -d "venv" ]; then
    log "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Step 2: Run training
log "${BLUE}Starting MAK training process...${NC}"
start_process "Training" "python src/learn.py" "logs/training.log"

# Step 3: Launch Feedback Dashboard
log "${BLUE}Launching Feedback Dashboard...${NC}"
start_process "Feedback Dashboard" "python src/mak_learning_pipeline.py" "logs/feedback.log"

# Step 4: Launch Streamlit Dashboard (optional)
if [ -f "src/ollo.py" ]; then
    log "${BLUE}Launching MAK Dashboard...${NC}"
    start_process "MAK Dashboard" "streamlit run src/ollo.py" "logs/dashboard.log"
fi

# Step 5: Monitor logs
log "${GREEN}MAK System is up and running!${NC}"
echo -e "${YELLOW}Services:${NC}"
echo -e "  - Training Log: ${GREEN}logs/training.log${NC}"
echo -e "  - Feedback Dashboard: ${GREEN}http://localhost:5000${NC}"
if [ -f "src/ollo.py" ]; then
    echo -e "  - MAK Dashboard: ${GREEN}http://localhost:8501${NC}"
fi

# Keep script running and monitor processes
while true; do
    if ! check_process "python src/learn.py"; then
        log "${RED}Training process stopped unexpectedly${NC}"
        break
    fi
    
    if ! check_process "python src/mak_learning_pipeline.py"; then
        log "${RED}Feedback Dashboard stopped unexpectedly${NC}"
        break
    fi
    
    if [ -f "src/ollo.py" ] && ! check_process "streamlit run src/ollo.py"; then
        log "${RED}MAK Dashboard stopped unexpectedly${NC}"
        break
    fi
    
    sleep 10
done

# Cleanup on exit
log "${YELLOW}Shutting down MAK System...${NC}"
pkill -f "python src/learn.py"
pkill -f "python src/mak_learning_pipeline.py"
pkill -f "streamlit run src/ollo.py"
log "${GREEN}MAK System shutdown complete${NC}" 