import os
import torch
import wandb
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from accelerate import Accelerator
from datasets import Dataset, load_dataset, concatenate_datasets
import deepspeed
from src.ai_model import (
    GRPO,
    Qwen2_5,
    Mistral7B,
    DeepSeek,
    SearchHandler
)
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MakReSearchTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the Mak ReSearch training system."""
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize Weights & Biases
        self._init_wandb()
        
        # Setup directories
        self._setup_directories()
        
        # Initialize models and components
        self._init_models()
        self._init_search_handler()
        
        # Track metrics
        self.best_f1 = 0.0
        self.best_model_path = None

    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        self.run = wandb.init(
            project="mak-research",
            name=f"mak-trainer-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=self.config
        )

    def _setup_directories(self):
        """Create necessary directories for the project."""
        for dir_name in ["results", "logs", "models", "data", "checkpoints"]:
            Path(f"./{dir_name}").mkdir(exist_ok=True)

    def _init_models(self):
        """Initialize the base models and tokenizer."""
        logger.info("Initializing models and tokenizer...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model"]["tokenizer"],
            trust_remote_code=True
        )
        
        # Initialize base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config["model"]["base_model"],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config["model"]["use_bfloat16"] else torch.float16,
            device_map=self.config["model"]["device_map"]
        )
        
        # Initialize specialized models
        self.qwen = Qwen2_5.from_pretrained(self.config["model"]["qwen_path"])
        self.mistral = Mistral7B.from_pretrained(self.config["model"]["mistral_path"])
        self.deepseek = DeepSeek.from_pretrained(self.config["model"]["deepseek_path"])

    def _init_search_handler(self):
        """Initialize the search handler for embedding and keyword search fusion."""
        logger.info("Initializing search handler...")
        self.search_handler = SearchHandler(
            embedding_model=self.config["search"]["embedding_model"],
            keyword_model=self.config["search"]["keyword_model"],
            fusion_strategy=self.config["search"]["fusion_strategy"],
            embedding_weight=self.config["search"]["embedding_weight"],
            keyword_weight=self.config["search"]["keyword_weight"]
        )
        
        # Load and index documents for search
        logger.info("Indexing documents for search...")
        documents = []
        
        # Load MuSiQue documents
        musique_path = self.config["dataset"]["musique"]["path"]
        if os.path.exists(musique_path):
            with open(musique_path, 'r') as f:
                musique_data = json.load(f)
                documents.extend([item["text"] for item in musique_data])
                
        # Load custom documents
        custom_path = self.config["dataset"]["custom"]["path"]
        if os.path.exists(custom_path):
            with open(custom_path, 'r') as f:
                custom_data = json.load(f)
                documents.extend([item["text"] for item in custom_data])
                
        if documents:
            self.search_handler.index_documents(documents)
        else:
            logger.warning("No documents found to index for search")

    def _prepare_research_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Prepare a ReSearch-format prompt with think-search-result-answer structure."""
        prompt_template = f"""<think>
Analyze the query and determine what information is needed to answer it.
</think>

<search>
{query}
</search>

<r>
{context if context else "No context provided"}
</r>

<answer>
Based on the search results, provide a comprehensive answer.
</answer>"""
        return prompt_template

    def _compute_grpo_loss(self, 
                          logits: torch.Tensor,
                          labels: torch.Tensor,
                          search_results: List[str],
                          attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute the GRPO loss with search result masking."""
        # Implement GRPO loss calculation
        pass

    def train(self):
        """Main training loop for the Mak ReSearch system."""
        logger.info("Starting training...")
        
        # Load and prepare dataset
        dataset = self._load_dataset()
        
        # Initialize GRPO trainer
        grpo_trainer = GRPO(
            model=self.base_model,
            tokenizer=self.tokenizer,
            config=self.config["grpo"]
        )
        
        # Training loop
        for epoch in range(self.config["training"]["num_epochs"]):
            logger.info(f"Starting epoch {epoch + 1}")
            
            # Training step
            for batch in dataset:
                # Generate search queries
                search_queries = self._generate_search_queries(batch)
                
                # Retrieve relevant information
                search_results = self.search_handler.batch_search(
                    search_queries,
                    top_k=self.config["search"]["top_k"],
                    min_score=self.config["search"]["min_score"]
                )
                
                # Prepare ReSearch prompts
                prompts = [self._prepare_research_prompt(q, r) for q, r in zip(search_queries, search_results)]
                
                # Generate responses
                outputs = grpo_trainer.generate(
                    prompts,
                    max_length=self.config["generation"]["max_length"],
                    num_return_sequences=1
                )
                
                # Compute loss and update
                loss, metrics = self._compute_grpo_loss(
                    outputs.logits,
                    batch["labels"],
                    search_results,
                    batch["attention_mask"]
                )
                
                # Log metrics
                wandb.log(metrics)
                
                # Save best model
                if metrics["f1"] > self.best_f1:
                    self._save_best_model(metrics["f1"])
            
            # Evaluate
            eval_metrics = self._evaluate()
            logger.info(f"Epoch {epoch + 1} evaluation metrics: {eval_metrics}")
            
            # Early stopping check
            if self._should_stop(eval_metrics):
                logger.info("Early stopping triggered")
                break
        
        logger.info("Training complete!")

    def _load_dataset(self) -> Dataset:
        """Load and prepare the training dataset."""
        logger.info("Loading dataset...")
        
        # Load MuSiQue dataset
        musique_path = self.config["dataset"]["musique"]["path"]
        if os.path.exists(musique_path):
            musique_dataset = load_dataset(
                "json",
                data_files=musique_path,
                split=self.config["dataset"]["musique"]["split"]
            )
        else:
            logger.warning(f"MuSiQue dataset not found at {musique_path}")
            musique_dataset = None
            
        # Load custom dataset
        custom_path = self.config["dataset"]["custom"]["path"]
        if os.path.exists(custom_path):
            custom_dataset = load_dataset(
                "json",
                data_files=custom_path,
                split="train"
            )
        else:
            logger.warning(f"Custom dataset not found at {custom_path}")
            custom_dataset = None
            
        # Combine datasets if available
        if musique_dataset and custom_dataset:
            dataset = concatenate_datasets([musique_dataset, custom_dataset])
        elif musique_dataset:
            dataset = musique_dataset
        elif custom_dataset:
            dataset = custom_dataset
        else:
            raise ValueError("No dataset found. Please provide either MuSiQue or custom dataset.")
            
        # Apply preprocessing
        dataset = dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                max_length=self.config["dataset"]["preprocessing"]["max_length"],
                padding=self.config["dataset"]["preprocessing"]["padding"],
                truncation=self.config["dataset"]["preprocessing"]["truncation"],
                return_tensors=self.config["dataset"]["preprocessing"]["return_tensors"]
            ),
            batched=True
        )
        
        # Set format for PyTorch
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        return dataset

    def _generate_search_queries(self, batch: Dict) -> List[str]:
        """Generate search queries from batch data."""
        # Decode input IDs to text
        texts = self.tokenizer.batch_decode(
            batch["input_ids"],
            skip_special_tokens=True
        )
        
        # Extract questions from the text
        queries = []
        for text in texts:
            # Split on question mark to get the question part
            parts = text.split("?")
            if len(parts) > 1:
                query = parts[0] + "?"
            else:
                # If no question mark, use the first sentence
                query = text.split(".")[0] + "?"
            queries.append(query)
            
        return queries

    def _evaluate(self) -> Dict:
        """Evaluate the model on validation data."""
        # Implement evaluation
        pass

    def _should_stop(self, metrics: Dict) -> bool:
        """Check early stopping conditions."""
        # Implement early stopping logic
        pass

    def _save_best_model(self, f1_score: float):
        """Save the best model checkpoint."""
        # Implement model saving
        pass

if __name__ == "__main__":
    trainer = MakReSearchTrainer()
    trainer.train() 