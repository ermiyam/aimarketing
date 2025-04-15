"""
Creates smart variations of prompts to improve Mak's understanding across topics.
"""

import os
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
import random
import re

class PromptOptimizer:
    def __init__(self, config_path: str = "config.ps1"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.prompt_history = []
        self.templates = self._load_templates()
        
        logging.info("Prompt optimizer initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        # TODO: Implement config loading
        return {}

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/optimizer_prompts.log'),
                logging.StreamHandler()
            ]
        )

    def _load_templates(self) -> List[Dict]:
        """Load prompt templates"""
        templates_path = "templates/prompt_templates.json"
        if os.path.exists(templates_path):
            with open(templates_path, "r") as f:
                return json.load(f)
        return []

    def generate_variations(self, base_prompt: str, num_variations: int = 3) -> List[str]:
        """Generate variations of a base prompt"""
        try:
            variations = []
            
            for _ in range(num_variations):
                # Apply different variation techniques
                variation = self._apply_variation_techniques(base_prompt)
                variations.append(variation)
                
                # Record the variation
                self._record_prompt_variation(base_prompt, variation)
            
            return variations
            
        except Exception as e:
            logging.error(f"Error generating prompt variations: {str(e)}")
            raise

    def _apply_variation_techniques(self, prompt: str) -> str:
        """Apply different techniques to vary the prompt"""
        # 1. Add context
        if random.random() < 0.3:
            prompt = self._add_context(prompt)
            
        # 2. Change perspective
        if random.random() < 0.3:
            prompt = self._change_perspective(prompt)
            
        # 3. Add constraints
        if random.random() < 0.3:
            prompt = self._add_constraints(prompt)
            
        # 4. Use template
        if self.templates and random.random() < 0.3:
            prompt = self._apply_template(prompt)
            
        return prompt

    def _add_context(self, prompt: str) -> str:
        """Add contextual information to the prompt"""
        contexts = [
            "Given the current state of technology,",
            "Considering recent developments in AI,",
            "In the context of modern machine learning,",
            "With respect to current industry standards,"
        ]
        return f"{random.choice(contexts)} {prompt}"

    def _change_perspective(self, prompt: str) -> str:
        """Change the perspective of the prompt"""
        perspectives = [
            "How would an expert in the field approach",
            "What would a beginner need to know about",
            "From a technical standpoint, explain",
            "In simple terms, describe"
        ]
        return f"{random.choice(perspectives)} {prompt}"

    def _add_constraints(self, prompt: str) -> str:
        """Add constraints to the prompt"""
        constraints = [
            "while keeping the explanation concise",
            "focusing on practical applications",
            "with specific examples",
            "in a step-by-step manner"
        ]
        return f"{prompt} {random.choice(constraints)}"

    def _apply_template(self, prompt: str) -> str:
        """Apply a template to the prompt"""
        template = random.choice(self.templates)
        return template["format"].format(prompt=prompt)

    def _record_prompt_variation(self, base_prompt: str, variation: str):
        """Record a prompt variation"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "base_prompt": base_prompt,
            "variation": variation,
            "techniques_used": self._analyze_variation_techniques(base_prompt, variation)
        }
        self.prompt_history.append(record)
        
        # Save to file
        with open("logs/prompt_history.json", "w") as f:
            json.dump(self.prompt_history, f, indent=2)

    def _analyze_variation_techniques(self, base: str, variation: str) -> List[str]:
        """Analyze which variation techniques were used"""
        techniques = []
        
        if len(variation) > len(base):
            techniques.append("context_addition")
            
        if re.search(r'\b(how|what|explain|describe)\b', variation.lower()):
            techniques.append("perspective_change")
            
        if re.search(r'\b(while|focusing|with|in)\b', variation.lower()):
            techniques.append("constraint_addition")
            
        return techniques

    def get_prompt_history(self) -> List[Dict]:
        """Get the history of prompt variations"""
        return self.prompt_history

if __name__ == "__main__":
    optimizer = PromptOptimizer()
    # Example usage
    # variations = optimizer.generate_variations("Explain machine learning")
    # print(variations)
