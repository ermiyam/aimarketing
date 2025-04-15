#!/bin/bash

# Navigate to your project directory
cd ~/mak

# Activate virtual environment
source venv/bin/activate

# Run torch distributed training on local node (rank 0)
torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=213.173.105.9 \
  --master_port=29500 \
  train_distributed.py 
