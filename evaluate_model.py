import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import jsonlines
import os

def load_model(model_path):
    """Load the trained model and tokenizer."""
    # Convert relative path to absolute path
    model_path = os.path.abspath(model_path)
    print(f"Loading model from: {model_path}")
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    return model, tokenizer

def prepare_test_data(tokenizer):
    """Prepare test data for evaluation."""
    # Load test data
    test_data = []
    with jsonlines.open('combined_learning_data/combined_marketing_learning_database.jsonl') as reader:
        for obj in reader:
            test_data.append(obj)
    
    # Extract features and labels
    texts = []
    labels = []
    for item in test_data:
        if 'failure_topic' in item and 'subtopics' in item:
            text = f"{item['failure_topic']} {' '.join(item['subtopics'])}"
            texts.append(text)
            labels.append(item['subtopics'][0])
    
    # Tokenize texts
    encodings = tokenizer(texts, truncation=True, padding=True)
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    
    return dataset, labels

def evaluate_model(model, tokenizer, dataset, true_labels):
    """Evaluate the model's performance."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for item in dataset:
            input_ids = torch.tensor([item['input_ids']]).to(device)
            attention_mask = torch.tensor([item['attention_mask']]).to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)
    
    # Generate classification report
    report = classification_report(true_labels, predictions, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    return report, cm

def save_evaluation_results(report, cm, output_dir):
    """Save evaluation results and visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save classification report
    with open(output_dir / "classification_report.txt", "w") as f:
        f.write("Classification Report:\n")
        for key, value in report.items():
            if isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k, v in value.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

def main():
    print("Loading model...")
    model, tokenizer = load_model("./models/marketing_failure_predictor")
    
    print("Preparing test data...")
    test_dataset, true_labels = prepare_test_data(tokenizer)
    
    print("Evaluating model...")
    report, cm = evaluate_model(model, tokenizer, test_dataset, true_labels)
    
    print("Saving evaluation results...")
    save_evaluation_results(report, cm, "evaluation_results")
    
    print("Evaluation complete! Check the 'evaluation_results' directory for results.")

if __name__ == "__main__":
    main() 