import torch
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging
from pathlib import Path
import json
from tqdm import tqdm

class SearchHandler:
    def __init__(self,
                 embedding_model: str = "BAAI/bge-large-en-v1.5",
                 keyword_model: str = "BM25",
                 fusion_strategy: str = "weighted",
                 embedding_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        """Initialize the search handler with embedding and keyword models.
        
        Args:
            embedding_model: Name of the sentence transformer model
            keyword_model: Type of keyword model to use
            fusion_strategy: Strategy for combining scores
            embedding_weight: Weight for embedding scores
            keyword_weight: Weight for keyword scores
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.keyword_model = keyword_model
        self.fusion_strategy = fusion_strategy
        self.embedding_weight = embedding_weight
        self.keyword_weight = keyword_weight
        
        # Initialize BM25 if needed
        if keyword_model == "BM25":
            self.bm25 = None
            self.corpus = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _tokenize_corpus(self, corpus: List[str]) -> List[List[str]]:
        """Tokenize corpus for BM25."""
        return [doc.lower().split() for doc in corpus]

    def build_index(self, corpus: List[str]):
        """Build search index from corpus.
        
        Args:
            corpus: List of documents to index
        """
        self.logger.info("Building search index...")
        
        # Store corpus
        self.corpus = corpus
        
        # Build BM25 index if needed
        if self.keyword_model == "BM25":
            tokenized_corpus = self._tokenize_corpus(corpus)
            self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Compute embeddings
        self.logger.info("Computing embeddings...")
        self.embeddings = self.embedding_model.encode(
            corpus,
            show_progress_bar=True,
            convert_to_tensor=True
        )
        
        self.logger.info("Index built successfully")

    def search(self,
              query: str,
              top_k: int = 5,
              threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Search corpus using both embedding and keyword models.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum score threshold
        
        Returns:
            results: List of (document, score) tuples
        """
        # Compute query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_tensor=True
        )
        
        # Compute embedding scores
        embedding_scores = torch.matmul(
            query_embedding,
            self.embeddings.T
        ).cpu().numpy()
        
        # Compute keyword scores if using BM25
        if self.keyword_model == "BM25":
            tokenized_query = query.lower().split()
            keyword_scores = self.bm25.get_scores(tokenized_query)
        else:
            keyword_scores = np.zeros(len(self.corpus))
        
        # Normalize scores
        embedding_scores = (embedding_scores - embedding_scores.min()) / (
            embedding_scores.max() - embedding_scores.min() + 1e-10
        )
        keyword_scores = (keyword_scores - keyword_scores.min()) / (
            keyword_scores.max() - keyword_scores.min() + 1e-10
        )
        
        # Combine scores based on fusion strategy
        if self.fusion_strategy == "weighted":
            combined_scores = (
                self.embedding_weight * embedding_scores +
                self.keyword_weight * keyword_scores
            )
        elif self.fusion_strategy == "max":
            combined_scores = np.maximum(embedding_scores, keyword_scores)
        elif self.fusion_strategy == "min":
            combined_scores = np.minimum(embedding_scores, keyword_scores)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        # Get top results
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        results = [
            (self.corpus[i], combined_scores[i])
            for i in top_indices
            if combined_scores[i] >= threshold
        ]
        
        return results

    def save_index(self, path: str):
        """Save search index to disk.
        
        Args:
            path: Path to save index
        """
        self.logger.info(f"Saving index to {path}...")
        
        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save corpus and embeddings
        torch.save({
            "corpus": self.corpus,
            "embeddings": self.embeddings,
            "bm25": self.bm25 if self.keyword_model == "BM25" else None,
            "config": {
                "embedding_model": self.embedding_model.get_name(),
                "keyword_model": self.keyword_model,
                "fusion_strategy": self.fusion_strategy,
                "embedding_weight": self.embedding_weight,
                "keyword_weight": self.keyword_weight
            }
        }, path)
        
        self.logger.info("Index saved successfully")

    def load_index(self, path: str):
        """Load search index from disk.
        
        Args:
            path: Path to load index from
        """
        self.logger.info(f"Loading index from {path}...")
        
        # Load index
        index = torch.load(path)
        
        # Restore state
        self.corpus = index["corpus"]
        self.embeddings = index["embeddings"]
        if self.keyword_model == "BM25":
            self.bm25 = index["bm25"]
        
        # Restore config
        config = index["config"]
        self.embedding_model = SentenceTransformer(config["embedding_model"])
        self.keyword_model = config["keyword_model"]
        self.fusion_strategy = config["fusion_strategy"]
        self.embedding_weight = config["embedding_weight"]
        self.keyword_weight = config["keyword_weight"]
        
        self.logger.info("Index loaded successfully")

class SearchResultCache:
    def __init__(self, cache_path: str = "./cache/search_results.json"):
        """Initialize search result cache.
        
        Args:
            cache_path: Path to cache file
        """
        self.cache_path = cache_path
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk."""
        if Path(self.cache_path).exists():
            with open(self.cache_path, 'r') as f:
                self.cache = json.load(f)

    def _save_cache(self):
        """Save cache to disk."""
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f)

    def get(self, query: str) -> Optional[List[Tuple[str, float]]]:
        """Get cached results for query.
        
        Args:
            query: Search query
        
        Returns:
            results: Cached results if available
        """
        return self.cache.get(query)

    def set(self, query: str, results: List[Tuple[str, float]]):
        """Cache results for query.
        
        Args:
            query: Search query
            results: Search results
        """
        self.cache[query] = results
        self._save_cache()

    def clear(self):
        """Clear the cache."""
        self.cache = {}
        self._save_cache() 