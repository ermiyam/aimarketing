import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import jsonlines
from pathlib import Path
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    """Load and prepare the training data."""
    data = []
    with jsonlines.open('combined_learning_data/combined_marketing_learning_database.jsonl') as reader:
        for obj in reader:
            data.append(obj)
    
    # Extract features and labels
    texts = []
    labels = []
    for item in data:
        if 'failure_topic' in item and 'subtopics' in item:
            text = f"{item['failure_topic']} {' '.join(item['subtopics'])}"
            texts.append(text)
            labels.append(item['subtopics'][0])
    
    return texts, labels

def prepare_dataset(texts, labels, tokenizer):
    """Prepare the dataset for training."""
    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Tokenize texts
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })
    
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

def main():
    # Create model directory if it doesn't exist
    model_dir = Path("./models/marketing_failure_predictor")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    texts, labels = load_data()
    
    print("Initializing tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(set(labels))
    )
    
    print("Preparing datasets...")
    train_dataset, val_dataset = prepare_dataset(texts, labels, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./training_output",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    print("Training complete! Model saved to:", model_dir)

if __name__ == "__main__":
    main() 