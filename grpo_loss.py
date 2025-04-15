import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from transformers import PreTrainedModel
from flashrag import FlashRAG

class GRPOLoss(nn.Module):
    def __init__(self, 
                 model: PreTrainedModel,
                 retriever: FlashRAG,
                 alpha: float = 0.1,
                 beta: float = 0.5,
                 gamma: float = 0.3,
                 max_kl: float = 0.1,
                 clip_range: float = 0.2,
                 value_clip_range: float = 0.2,
                 entropy_coef: float = 0.01):
        """Initialize GRPO loss with search result masking.
        
        Args:
            model: The base language model
            retriever: The search retriever
            alpha: Weight for search result masking
            beta: Weight for policy gradient loss
            gamma: Weight for value loss
            max_kl: Maximum KL divergence
            clip_range: PPO clip range
            value_clip_range: Value function clip range
            entropy_coef: Entropy coefficient
        """
        super().__init__()
        self.model = model
        self.retriever = retriever
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_kl = max_kl
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.entropy_coef = entropy_coef

    def _mask_search_results(self, 
                           logits: torch.Tensor,
                           search_results: List[str],
                           attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply search result masking to logits."""
        # Tokenize search results
        search_tokens = self.retriever.tokenize(search_results)
        
        # Create mask tensor
        mask = torch.zeros_like(logits, dtype=torch.bool)
        
        # Apply mask for search result tokens
        for i, tokens in enumerate(search_tokens):
            for token in tokens:
                mask[i, :, token] = True
        
        # Apply mask to logits
        masked_logits = logits.masked_fill(~mask, float('-inf'))
        
        return masked_logits

    def _compute_kl_divergence(self,
                             old_logits: torch.Tensor,
                             new_logits: torch.Tensor,
                             attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between old and new policy."""
        old_probs = F.softmax(old_logits, dim=-1)
        new_probs = F.softmax(new_logits, dim=-1)
        
        kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=-1)
        kl = (kl * attention_mask).sum() / attention_mask.sum()
        
        return kl

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the policy."""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropy.mean()

    def forward(self,
                old_logits: torch.Tensor,
                new_logits: torch.Tensor,
                values: torch.Tensor,
                returns: torch.Tensor,
                advantages: torch.Tensor,
                search_results: List[str],
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute the GRPO loss with search result masking.
        
        Args:
            old_logits: Logits from old policy
            new_logits: Logits from new policy
            values: Value function predictions
            returns: Monte Carlo returns
            advantages: Advantage estimates
            search_results: Retrieved search results
            attention_mask: Attention mask for padding
        
        Returns:
            loss: Total loss
            metrics: Dictionary of metrics
        """
        # Apply search result masking
        masked_new_logits = self._mask_search_results(
            new_logits, search_results, attention_mask
        )
        
        # Compute policy loss
        ratio = torch.exp(
            F.log_softmax(masked_new_logits, dim=-1) - 
            F.log_softmax(old_logits, dim=-1)
        )
        policy_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        ).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values, returns)
        value_loss = torch.clamp(
            value_loss,
            -self.value_clip_range,
            self.value_clip_range
        )
        
        # Compute entropy bonus
        entropy = self._compute_entropy(masked_new_logits)
        
        # Compute KL divergence
        kl = self._compute_kl_divergence(
            old_logits, masked_new_logits, attention_mask
        )
        
        # Compute total loss
        total_loss = (
            self.beta * policy_loss +
            self.gamma * value_loss -
            self.entropy_coef * entropy +
            self.alpha * kl
        )
        
        # Compute metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "kl_divergence": kl.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, metrics

class GRPOTrainer:
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 retriever: FlashRAG,
                 config: Dict):
        """Initialize GRPO trainer.
        
        Args:
            model: The base language model
            tokenizer: The tokenizer
            retriever: The search retriever
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.config = config
        
        self.loss_fn = GRPOLoss(
            model=model,
            retriever=retriever,
            alpha=config["alpha"],
            beta=config["beta"],
            gamma=config["gamma"],
            max_kl=config["max_kl"],
            clip_range=config["clip_range"],
            value_clip_range=config["value_clip_range"],
            entropy_coef=config["entropy_coef"]
        )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

    def train_step(self,
                  batch: Dict,
                  search_results: List[str]) -> Tuple[torch.Tensor, Dict]:
        """Perform a single training step.
        
        Args:
            batch: Input batch
            search_results: Retrieved search results
        
        Returns:
            loss: Training loss
            metrics: Training metrics
        """
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        # Compute loss
        loss, metrics = self.loss_fn(
            old_logits=outputs.logits.detach(),
            new_logits=outputs.logits,
            values=outputs.value_preds,
            returns=batch["returns"],
            advantages=batch["advantages"],
            search_results=search_results,
            attention_mask=batch["attention_mask"]
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config["max_grad_norm"]
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss, metrics

    def generate(self,
                prompts: List[str],
                max_length: int,
                num_return_sequences: int = 1) -> Dict:
        """Generate responses with search result masking.
        
        Args:
            prompts: Input prompts
            max_length: Maximum generation length
            num_return_sequences: Number of sequences to generate
        
        Returns:
            outputs: Generation outputs
        """
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate with search result masking
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            temperature=self.config["temperature"],
            top_p=self.config["top_p"]
        )
        
        return outputs 