import os
import json
import torch
import logging
from datetime import datetime
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Using model: {config['model']['base_model']}")

    # Initialize model and tokenizer
    logger.info("Loading model and tokenizer...")
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

    # Load and prepare dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files="mak_dataset.json")
    logger.info(f"Loaded {len(dataset['train'])} training examples")

    # Tokenize function with labels
    def tokenize(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config['training']['max_seq_len'],
            return_tensors=None
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    # Prepare dataset
    train_dataset = dataset["train"].map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

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
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim='adamw_torch',
        remove_unused_columns=False,
        report_to="none",  # Disable wandb for now
        logging_dir="./logs",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model("./final_model")
    logger.info("Training completed and model saved!")

if __name__ == "__main__":
    main() 