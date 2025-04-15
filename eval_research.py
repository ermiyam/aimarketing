import os
import torch
import wandb
import json
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset, load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ResearchEvaluator:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 search_handler,
                 config: Dict):
        """Initialize the evaluator.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer
            search_handler: The search handler
            config: Evaluation configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.search_handler = search_handler
        self.config = config
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

    def _prepare_research_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Prepare a ReSearch-format prompt."""
        prompt_template = f"""<think>
Analyze the query and determine what information is needed to answer it.
</think>

<search>
{query}
</search>

<result>
{context if context else "No context provided"}
</result>

<answer>
Based on the search results, provide a comprehensive answer.
</answer>"""
        return prompt_template

    def _generate_response(self,
                         prompt: str,
                         max_length: int = 2048) -> str:
        """Generate response from model."""
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"]
        )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return response

    def _extract_answer(self, response: str) -> str:
        """Extract answer from response."""
        # Find answer section
        start_idx = response.find("<answer>")
        end_idx = response.find("</answer>")
        
        if start_idx == -1 or end_idx == -1:
            return response
        
        return response[start_idx + 8:end_idx].strip()

    def _compute_metrics(self,
                        predictions: List[str],
                        references: List[str]) -> Dict:
        """Compute evaluation metrics.
        
        Args:
            predictions: Model predictions
            references: Reference answers
        
        Returns:
            metrics: Dictionary of metrics
        """
        # Initialize metrics
        metrics = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "bleu": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": []
        }
        
        # Compute metrics for each example
        for pred, ref in zip(predictions, references):
            # ROUGE scores
            scores = self.scorer.score(ref, pred)
            metrics["rouge1"].append(scores["rouge1"].fmeasure)
            metrics["rouge2"].append(scores["rouge2"].fmeasure)
            metrics["rougeL"].append(scores["rougeL"].fmeasure)
            
            # BLEU score
            bleu = sentence_bleu(
                [ref.split()],
                pred.split(),
                weights=(0.25, 0.25, 0.25, 0.25)
            )
            metrics["bleu"].append(bleu)
            
            # Token-level metrics
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            # Accuracy
            accuracy = len(pred_tokens & ref_tokens) / len(ref_tokens)
            metrics["accuracy"].append(accuracy)
            
            # F1, precision, recall
            if len(pred_tokens) > 0:
                precision = len(pred_tokens & ref_tokens) / len(pred_tokens)
                recall = len(pred_tokens & ref_tokens) / len(ref_tokens)
                f1 = 2 * precision * recall / (precision + recall)
            else:
                precision = recall = f1 = 0.0
            
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
        
        # Average metrics
        avg_metrics = {
            k: np.mean(v) for k, v in metrics.items()
        }
        
        return avg_metrics

    def evaluate(self,
                dataset: Dataset,
                output_path: Optional[str] = None) -> Dict:
        """Evaluate model on dataset.
        
        Args:
            dataset: Evaluation dataset
            output_path: Path to save results
        
        Returns:
            metrics: Evaluation metrics
        """
        self.logger.info("Starting evaluation...")
        
        # Initialize lists
        predictions = []
        references = []
        search_queries = []
        search_results = []
        
        # Evaluate each example
        for example in tqdm(dataset):
            # Get query and reference
            query = example["question"]
            reference = example["answer"]
            
            # Search for relevant information
            results = self.search_handler.search(query)
            context = "\n".join([r[0] for r in results])
            
            # Prepare prompt
            prompt = self._prepare_research_prompt(query, context)
            
            # Generate response
            response = self._generate_response(prompt)
            answer = self._extract_answer(response)
            
            # Store results
            predictions.append(answer)
            references.append(reference)
            search_queries.append(query)
            search_results.append(context)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, references)
        
        # Save results if path provided
        if output_path:
            self._save_results(
                output_path,
                predictions,
                references,
                search_queries,
                search_results,
                metrics
            )
        
        return metrics

    def _save_results(self,
                     path: str,
                     predictions: List[str],
                     references: List[str],
                     search_queries: List[str],
                     search_results: List[str],
                     metrics: Dict):
        """Save evaluation results.
        
        Args:
            path: Path to save results
            predictions: Model predictions
            references: Reference answers
            search_queries: Search queries
            search_results: Retrieved results
            metrics: Evaluation metrics
        """
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results
        results = {
            "metrics": metrics,
            "examples": [
                {
                    "query": q,
                    "prediction": p,
                    "reference": r,
                    "search_results": s
                }
                for q, p, r, s in zip(
                    search_queries,
                    predictions,
                    references,
                    search_results
                )
            ]
        }
        
        # Save to file
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {path}")

if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from search_handler import SearchHandler
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    
    # Initialize search handler
    search_handler = SearchHandler()
    
    # Load evaluation dataset
    dataset = load_dataset("musique", split="validation")
    
    # Initialize evaluator
    evaluator = ResearchEvaluator(
        model=model,
        tokenizer=tokenizer,
        search_handler=search_handler,
        config={
            "temperature": 0.7,
            "top_p": 0.9
        }
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(
        dataset,
        output_path="./results/evaluation.json"
    )
    
    # Print metrics
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}") 