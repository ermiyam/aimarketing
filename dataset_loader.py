"""
Downloads and prepares HuggingFace datasets like OpenCodeReasoning, LLaVA, Nemotron, etc.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import asyncio
from tqdm import tqdm
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import torch

class DatasetLoader:
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 musique_path: str = "./data/musique",
                 fineweb_path: str = "./data/fineweb",
                 youtube_path: str = "./data/youtube_transcripts",
                 wikipedia_path: str = "./data/wikipedia",
                 batch_size: int = 8,
                 num_workers: int = 4):
        """Initialize the dataset loader.
        
        Args:
            tokenizer: Tokenizer for processing text
            musique_path: Path to MuSiQue dataset
            fineweb_path: Path to FineWeb dataset
            youtube_path: Path to YouTube transcripts
            wikipedia_path: Path to Wikipedia dump
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
        """
        self.tokenizer = tokenizer
        self.musique_path = musique_path
        self.fineweb_path = fineweb_path
        self.youtube_path = youtube_path
        self.wikipedia_path = wikipedia_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        self.api = HfApi()
        self.dataset_cache = {}
        self.wasabi_client = None
        self.nlp = None
        self._init_nlp()
        self._download_nltk_data()
        self.dataset_categories = {
            "reasoning": [
                "nvidia/OpenCodeReasoning",
                "nvidia/Llama-Nemotron-Post-Training-Dataset",
                "open-thoughts/OpenThoughts2-1M",
                "glaiveai/reasoning-v1-20m",
                "facebook/natural_reasoning",
                "davanstrien/reasoning-required",
                "allenai/dolma",
                "HuggingFaceFW/fineweb",
                "HuggingFaceFW/fineweb-2",
                "future-technologies/Universal-Transformers-Dataset",
                "Intelligent-Internet/II-Thought-RL-v0"
            ],
            "knowledge": [
                "wikimedia/wikipedia",
                "MaziyarPanahi/OpenCodeReasoning_ShareGPT",
                "yahma/alpaca-cleaned",
                "UCSC-VLAA/STAR-1"
            ],
            "instruction": [
                "openai/openai_humaneval",
                "liuhaotian/LLaVA-Instruct-150K",
                "oumi-ai/oumi-synthetic-document-claims"
            ],
            "emotions": [
                "google-research-datasets/go_emotions",
                "gaia-benchmark/GAIA",
                "ByteDance-Seed/Multi-SWE-bench"
            ]
        }
        self._load_cache()
        self._init_wasabi()
        self._init_tokenizer()
        logging.info("Dataset loader initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {str(e)}")
            return {}

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/dataset_loader.log'),
                logging.StreamHandler()
            ]
        )

    async def check_for_updates(self) -> List[str]:
        """Check for new or updated datasets"""
        try:
            updated_datasets = []
            for category, datasets in self.dataset_categories.items():
                for dataset_name in datasets:
                    if await self._is_dataset_updated(dataset_name):
                        updated_datasets.append(dataset_name)
                        logging.info(f"Dataset {dataset_name} has updates")
            return updated_datasets
        except Exception as e:
            logging.error(f"Error checking for dataset updates: {str(e)}")
            raise

    async def _is_dataset_updated(self, dataset_name: str) -> bool:
        """Check if a dataset has been updated"""
        try:
            dataset_info = self.api.dataset_info(dataset_name)
            if dataset_name in self.dataset_cache:
                cached_version = self.dataset_cache[dataset_name]
                return dataset_info.last_modified > cached_version
            return True
        except Exception as e:
            logging.error(f"Error checking dataset {dataset_name}: {str(e)}")
            return False

    async def load_dataset(self, dataset_name: str) -> Dataset:
        """Load a specific dataset"""
        try:
            logging.info(f"Loading dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)
            self.dataset_cache[dataset_name] = datetime.now().isoformat()
            self._save_cache()
            return dataset
        except Exception as e:
            logging.error(f"Error loading dataset {dataset_name}: {str(e)}")
            raise

    async def load_all_datasets(self) -> Dict[str, Dataset]:
        """Load all datasets from all categories"""
        all_datasets = {}
        for category, datasets in self.dataset_categories.items():
            logging.info(f"Loading {category} datasets...")
            for dataset_name in tqdm(datasets, desc=f"Loading {category}"):
                try:
                    dataset = await self.load_dataset(dataset_name)
                    all_datasets[dataset_name] = dataset
                except Exception as e:
                    logging.error(f"Error loading {dataset_name}: {str(e)}")
        return all_datasets

    async def sync_to_wasabi(self, dataset: Dataset, dataset_name: str):
        """Sync dataset to Wasabi storage"""
        try:
            # TODO: Implement Wasabi sync
            logging.info(f"Syncing {dataset_name} to Wasabi...")
            # Placeholder for Wasabi sync implementation
            pass
        except Exception as e:
            logging.error(f"Error syncing {dataset_name} to Wasabi: {str(e)}")
            raise

    def _save_cache(self):
        """Save dataset cache to file"""
        cache_path = "data/dataset_cache.json"
        os.makedirs("data", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(self.dataset_cache, f, indent=2)

    def _load_cache(self):
        """Load dataset cache from file"""
        cache_path = "data/dataset_cache.json"
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                self.dataset_cache = json.load(f)

    async def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset for training"""
        try:
            processed_dataset = dataset.map(
                self._preprocess_example,
                batched=True,
                remove_columns=dataset.column_names
            )
            return processed_dataset
        except Exception as e:
            logging.error(f"Error preprocessing dataset: {str(e)}")
            raise

    def _preprocess_example(self, example):
        """Preprocess a single example"""
        # TODO: Implement dataset-specific preprocessing
        return example

    def get_available_datasets(self) -> Dict[str, List[str]]:
        """Get list of available datasets by category"""
        return self.dataset_categories

    def get_dataset_category(self, dataset_name: str) -> Optional[str]:
        """Get the category of a specific dataset"""
        for category, datasets in self.dataset_categories.items():
            if dataset_name in datasets:
                return category
        return None

    def _init_nlp(self):
        """Initialize spaCy NLP model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logging.info("spaCy model initialized")
        except Exception as e:
            logging.error(f"Error initializing spaCy model: {str(e)}")
            self.nlp = None

    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            logging.info("NLTK data downloaded")
        except Exception as e:
            logging.error(f"Error downloading NLTK data: {str(e)}")

    def get_data_quality_metrics(self, dataset: Dataset) -> Dict[str, Any]:
        """Calculate comprehensive data quality metrics"""
        metrics = {
            'basic_stats': self._get_basic_stats(dataset),
            'text_quality': self._get_text_quality_metrics(dataset),
            'diversity': self._get_diversity_metrics(dataset),
            'consistency': self._get_consistency_metrics(dataset),
            'completeness': self._get_completeness_metrics(dataset)
        }
        return metrics

    def _get_basic_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """Calculate basic dataset statistics"""
        stats = {
            'num_examples': len(dataset),
            'columns': dataset.column_names,
            'size_bytes': dataset.nbytes if hasattr(dataset, 'nbytes') else None
        }
        
        if 'text' in dataset.column_names:
            text_lengths = [len(str(text)) for text in dataset['text']]
            stats.update({
                'avg_text_length': np.mean(text_lengths),
                'max_text_length': max(text_lengths),
                'min_text_length': min(text_lengths),
                'std_text_length': np.std(text_lengths)
            })
        
        return stats

    def _get_text_quality_metrics(self, dataset: Dataset) -> Dict[str, Any]:
        """Calculate text quality metrics"""
        if 'text' not in dataset.column_names:
            return {}
        
        texts = dataset['text']
        quality_metrics = {
            'avg_word_length': 0,
            'vocabulary_size': 0,
            'stopword_ratio': 0,
            'readability_score': 0,
            'grammar_errors': 0
        }
        
        all_words = []
        stop_words = set(stopwords.words('english'))
        total_stopwords = 0
        total_words = 0
        
        for text in texts:
            words = word_tokenize(str(text).lower())
            all_words.extend(words)
            total_words += len(words)
            total_stopwords += sum(1 for word in words if word in stop_words)
        
        if total_words > 0:
            quality_metrics.update({
                'avg_word_length': sum(len(word) for word in all_words) / total_words,
                'vocabulary_size': len(set(all_words)),
                'stopword_ratio': total_stopwords / total_words
            })
        
        return quality_metrics

    def _get_diversity_metrics(self, dataset: Dataset) -> Dict[str, Any]:
        """Calculate diversity metrics"""
        if 'text' not in dataset.column_names:
            return {}
        
        texts = dataset['text']
        diversity_metrics = {
            'unique_words_ratio': 0,
            'topic_diversity': 0,
            'semantic_diversity': 0
        }
        
        # Calculate unique words ratio
        all_words = []
        for text in texts:
            words = word_tokenize(str(text).lower())
            all_words.extend(words)
        
        if all_words:
            diversity_metrics['unique_words_ratio'] = len(set(all_words)) / len(all_words)
        
        # Calculate semantic diversity using TF-IDF
        if len(texts) > 1:
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            diversity_metrics['semantic_diversity'] = 1 - np.mean(similarity_matrix)
        
        return diversity_metrics

    def _get_consistency_metrics(self, dataset: Dataset) -> Dict[str, Any]:
        """Calculate consistency metrics"""
        if 'text' not in dataset.column_names:
            return {}
        
        texts = dataset['text']
        consistency_metrics = {
            'format_consistency': 0,
            'length_consistency': 0,
            'style_consistency': 0
        }
        
        # Calculate format consistency
        formats = []
        for text in texts:
            text = str(text)
            formats.append({
                'has_url': bool(re.search(r'http[s]?://', text)),
                'has_code': bool(re.search(r'```.*?```', text, re.DOTALL)),
                'has_markdown': bool(re.search(r'[#*_]', text))
            })
        
        format_counts = Counter(tuple(f.items()) for f in formats)
        consistency_metrics['format_consistency'] = max(format_counts.values()) / len(formats)
        
        # Calculate length consistency
        lengths = [len(str(text)) for text in texts]
        consistency_metrics['length_consistency'] = 1 - (np.std(lengths) / np.mean(lengths))
        
        return consistency_metrics

    def _get_completeness_metrics(self, dataset: Dataset) -> Dict[str, Any]:
        """Calculate completeness metrics"""
        completeness_metrics = {
            'missing_values': {},
            'field_completeness': {}
        }
        
        for column in dataset.column_names:
            missing_count = sum(1 for value in dataset[column] if pd.isna(value) or value == '')
            completeness_metrics['missing_values'][column] = missing_count
            completeness_metrics['field_completeness'][column] = 1 - (missing_count / len(dataset))
        
        return completeness_metrics

    def get_dataset_health_report(self, dataset: Dataset) -> Dict[str, Any]:
        """Generate a comprehensive health report for the dataset"""
        metrics = self.get_data_quality_metrics(dataset)
        
        health_report = {
            'overall_health_score': self._calculate_health_score(metrics),
            'issues': self._identify_issues(metrics),
            'recommendations': self._generate_recommendations(metrics),
            'metrics': metrics
        }
        
        return health_report

    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall health score from metrics"""
        weights = {
            'basic_stats': 0.2,
            'text_quality': 0.3,
            'diversity': 0.2,
            'consistency': 0.15,
            'completeness': 0.15
        }
        
        score = 0
        for category, weight in weights.items():
            if category in metrics:
                category_score = np.mean([v for v in metrics[category].values() if isinstance(v, (int, float))])
                score += category_score * weight
        
        return score

    def _identify_issues(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify potential issues in the dataset"""
        issues = []
        
        # Check completeness
        if 'completeness' in metrics:
            for field, completeness in metrics['completeness']['field_completeness'].items():
                if completeness < 0.9:
                    issues.append(f"Low completeness in field '{field}': {completeness:.2%}")
        
        # Check text quality
        if 'text_quality' in metrics:
            if metrics['text_quality']['stopword_ratio'] > 0.5:
                issues.append("High stopword ratio in text")
            if metrics['text_quality']['vocabulary_size'] < 1000:
                issues.append("Limited vocabulary size")
        
        # Check consistency
        if 'consistency' in metrics:
            if metrics['consistency']['format_consistency'] < 0.7:
                issues.append("Low format consistency")
            if metrics['consistency']['length_consistency'] < 0.7:
                issues.append("Low length consistency")
        
        return issues

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving dataset quality"""
        recommendations = []
        
        # Text quality recommendations
        if 'text_quality' in metrics:
            if metrics['text_quality']['stopword_ratio'] > 0.5:
                recommendations.append("Consider reducing stopwords in text preprocessing")
            if metrics['text_quality']['vocabulary_size'] < 1000:
                recommendations.append("Consider adding more diverse vocabulary")
        
        # Completeness recommendations
        if 'completeness' in metrics:
            for field, completeness in metrics['completeness']['field_completeness'].items():
                if completeness < 0.9:
                    recommendations.append(f"Improve data collection for field '{field}'")
        
        # Consistency recommendations
        if 'consistency' in metrics:
            if metrics['consistency']['format_consistency'] < 0.7:
                recommendations.append("Standardize text format across the dataset")
            if metrics['consistency']['length_consistency'] < 0.7:
                recommendations.append("Standardize text length across the dataset")
        
        return recommendations

    async def validate_dataset(self, dataset: Dataset) -> Tuple[bool, List[str]]:
        """Validate dataset against quality standards"""
        metrics = self.get_data_quality_metrics(dataset)
        health_score = self._calculate_health_score(metrics)
        issues = self._identify_issues(metrics)
        
        is_valid = health_score >= 0.7 and len(issues) < 3
        return is_valid, issues

    def _load_musique(self) -> Dataset:
        """Load and process MuSiQue dataset."""
        self.logger.info("Loading MuSiQue dataset...")
        
        # Load dataset
        dataset = load_dataset("musique", split="train")
        
        # Process examples
        def process_example(example):
            # Combine question and context
            text = f"Question: {example['question']}\nContext: {example['context']}"
            
            # Tokenize
            tokenized = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            return {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "labels": tokenized["input_ids"][0]
            }
        
        # Process dataset
        processed_dataset = dataset.map(
            process_example,
            remove_columns=dataset.column_names,
            num_proc=self.num_workers
        )
        
        return processed_dataset

    def _load_fineweb(self) -> Dataset:
        """Load and process FineWeb dataset."""
        self.logger.info("Loading FineWeb dataset...")
        
        # Load dataset
        dataset = load_dataset("fineweb", split="train")
        
        # Process examples
        def process_example(example):
            # Tokenize
            tokenized = self.tokenizer(
                example["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            return {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "labels": tokenized["input_ids"][0]
            }
        
        # Process dataset
        processed_dataset = dataset.map(
            process_example,
            remove_columns=dataset.column_names,
            num_proc=self.num_workers
        )
        
        return processed_dataset

    def _load_youtube_transcripts(self) -> Dataset:
        """Load and process YouTube transcripts."""
        self.logger.info("Loading YouTube transcripts...")
        
        # Load transcripts
        transcripts = []
        for file in Path(self.youtube_path).glob("*.json"):
            with open(file, "r") as f:
                transcripts.extend(json.load(f))
        
        # Create dataset
        dataset = Dataset.from_dict({
            "text": [t["text"] for t in transcripts]
        })
        
        # Process examples
        def process_example(example):
            # Tokenize
            tokenized = self.tokenizer(
                example["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            return {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "labels": tokenized["input_ids"][0]
            }
        
        # Process dataset
        processed_dataset = dataset.map(
            process_example,
            remove_columns=dataset.column_names,
            num_proc=self.num_workers
        )
        
        return processed_dataset

    def _load_wikipedia(self) -> Dataset:
        """Load and process Wikipedia dump."""
        self.logger.info("Loading Wikipedia dump...")
        
        # Load dataset
        dataset = load_dataset("wikipedia", "20220301.en", split="train")
        
        # Process examples
        def process_example(example):
            # Tokenize
            tokenized = self.tokenizer(
                example["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            return {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "labels": tokenized["input_ids"][0]
            }
        
        # Process dataset
        processed_dataset = dataset.map(
            process_example,
            remove_columns=dataset.column_names,
            num_proc=self.num_workers
        )
        
        return processed_dataset

    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """Load and combine all datasets.
        
        Returns:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        # Load individual datasets
        musique_dataset = self._load_musique()
        fineweb_dataset = self._load_fineweb()
        youtube_dataset = self._load_youtube_transcripts()
        wikipedia_dataset = self._load_wikipedia()
        
        # Combine datasets
        combined_dataset = Dataset.concatenate_datasets([
            musique_dataset,
            fineweb_dataset,
            youtube_dataset,
            wikipedia_dataset
        ])
        
        # Shuffle and split
        combined_dataset = combined_dataset.shuffle(seed=42)
        split_dataset = combined_dataset.train_test_split(test_size=0.1)
        
        return split_dataset["train"], split_dataset["test"]

    def get_dataloader(self,
                      dataset: Dataset,
                      is_training: bool = True) -> torch.utils.data.DataLoader:
        """Create a dataloader for the dataset.
        
        Args:
            dataset: Dataset to create dataloader for
            is_training: Whether the dataloader is for training
        
        Returns:
            dataloader: DataLoader for the dataset
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=is_training,
            num_workers=self.num_workers,
            pin_memory=True
        )

if __name__ == "__main__":
    async def main():
        loader = DatasetLoader()
        # Example usage
        updated = await loader.check_for_updates()
        if updated:
            print(f"Found {len(updated)} updated datasets")
            datasets = await loader.load_all_datasets()
            print(f"Loaded {len(datasets)} datasets")
            
    asyncio.run(main())
