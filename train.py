import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
import wandb
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scraper import MarketingCampaignScraper
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketingFailureTrainer:
    def __init__(self):
        # Initialize Weights & Biases with more detailed config
        self.run = wandb.init(
            project="marketing-failure-predictor",
            name=f"marketing-trainer-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "model_name": "bert-base-uncased",
                "max_length": 512,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "epochs": 5
            }
        )

        # Create necessary directories
        self.setup_directories()
        
        # Initialize model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Track best metrics
        self.best_f1 = 0.0
        self.best_model_path = None

    def setup_directories(self):
        """Create necessary directories for the project."""
        for dir_name in ["results", "logs", "models", "data"]:
            Path(f"./{dir_name}").mkdir(exist_ok=True)

    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer."""
        logger.info("Initializing model and tokenizer...")
        self.model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True
        )

    def scrape_new_data(self):
        """Scrape new marketing campaign data."""
        logger.info("Scraping new marketing campaign data...")
        scraper = MarketingCampaignScraper()
        return scraper.scrape_and_save()

    def load_and_prepare_data(self, include_scraped=True):
        """Load and prepare the marketing failure insights data."""
        data = []
        
        # Load existing data
        file_paths = ['failure_insights_batch_002.jsonl']
        
        # Add scraped data if available
        if include_scraped and os.path.exists('data/scraped_insights.jsonl'):
            file_paths.append('data/scraped_insights.jsonl')
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                logger.warning(f"Data file not found: {file_path}")
                continue
            
            logger.info(f"Loading data from {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(eval(line.strip()))
        
        # Convert to HuggingFace dataset format
        texts = []
        labels = []
        for item in data:
            if 'failure_topic' in item and 'subtopics' in item and item['subtopics']:
                text = f"{item['failure_topic']} {' '.join(item['subtopics'])}"
                texts.append(text)
                labels.append(1 if "bad targeting" in text.lower() else 0)
        
        return texts, labels

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        accuracy = accuracy_score(labels, predictions)
        
        # Track best F1 score
        if f1 > self.best_f1:
            self.best_f1 = f1
            # Save the best model
            if hasattr(self, 'trainer'):
                best_model_path = f"./models/best_model_f1_{f1:.4f}"
                self.trainer.save_model(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                self.best_model_path = best_model_path
                logger.info(f"New best model saved with F1 score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self):
        """Train the model with the prepared dataset."""
        # First, scrape new data
        self.scrape_new_data()
        
        # Load and prepare all data
        logger.info("Loading and preparing dataset...")
        texts, labels = self.load_and_prepare_data()
        
        # Create dataset
        logger.info("Creating dataset...")
        dataset = Dataset.from_dict({
            "text": texts, 
            "label": labels
        })
        
        # Tokenize the data
        logger.info("Tokenizing data...")
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ),
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Split into train and validation
        split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
        
        # Calculate optimal batch size
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            batch_size = min(16, max(4, int(gpu_memory / (1024**3) * 2)))
            logger.info(f"Using batch size of {batch_size} based on available GPU memory")
        else:
            batch_size = 8
            logger.info("No GPU detected, using conservative batch size of 8")
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            warmup_ratio=0.1,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            eval_steps=50,
            save_steps=50,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            save_total_limit=3,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="wandb",
            gradient_accumulation_steps=4,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=4 if os.name != 'nt' else 0,
            remove_unused_columns=True,
            group_by_length=True,
            logging_first_step=True,
            ddp_find_unused_parameters=False,
            seed=42,
            push_to_hub=False
        )
        
        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        logger.info("Starting training...")
        self.trainer.train()
        
        # Final evaluation
        logger.info("\nPerforming final evaluation...")
        eval_results = self.trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
        
        # Save the final model
        logger.info("\nSaving final model...")
        final_model_path = "./models/final_marketing_failure_predictor"
        self.trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        # Log best model information
        if self.best_model_path:
            logger.info(f"\nBest model saved at: {self.best_model_path}")
            logger.info(f"Best F1 score achieved: {self.best_f1:.4f}")
        
        logger.info("Training complete!")

if __name__ == "__main__":
    trainer = MarketingFailureTrainer()
    trainer.train()
