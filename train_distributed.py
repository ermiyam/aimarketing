import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import yaml

def setup_training():
    """Initialize training environment - handles both single GPU and distributed cases"""
    is_distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
    
    if is_distributed:
        # Distributed training setup
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if os.name == 'nt':  # Windows
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        
        # Initialize process group without libuv
        if not dist.is_initialized():
            dist.init_process_group(
                backend='gloo' if os.name == 'nt' else 'nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
    else:
        # Single GPU setup
        rank = 0
        local_rank = 0
        world_size = 1
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
        
    return device, rank, local_rank, world_size, is_distributed

def main():
    # Setup training environment
    device, rank, local_rank, world_size, is_distributed = setup_training()
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if rank == 0:
        print(f"Training setup:")
        print(f"- Distributed: {is_distributed}")
        print(f"- World size: {world_size}")
        print(f"- Rank: {rank}")
        print(f"- Local rank: {local_rank}")
        print(f"- Device: {device}")
        print(f"- Model: {config['model']['base_model']}")

    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['base_model'],
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['tokenizer'],
        trust_remote_code=True
    )

    # Move model to device
    model = model.to(device)
    
    # Wrap model with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        fp16=True,
        bf16=False,
        gradient_checkpointing=config['system']['gradient_checkpointing'],
        ddp_find_unused_parameters=False,
        local_rank=local_rank
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,  # Add your dataset here
        tokenizer=tokenizer
    )

    # Start training
    trainer.train()

    # Save final model
    if rank == 0:
        trainer.save_model("./final_model")
        print("Training completed and model saved!")

    # Cleanup
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 