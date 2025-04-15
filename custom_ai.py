import torch
import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
import json

class CustomAI:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-6.7b-base"):
        """
        Initialize the AI with custom parameters
        """
        self.knowledge_base = {}
        self.learning_history = []
        self.model_name = model_name
        
        # Initialize model and tokenizer
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Process user query with optional context
        """
        # Add to learning history
        self.learning_history.append({
            'query': query,
            'timestamp': time.time(),
            'context': context
        })

        # Generate response
        response = self.generate_response(query, context)
        
        # Store in knowledge base
        self.update_knowledge(query, response)
        
        return {
            'response': response,
            'source': 'model',
            'timestamp': time.time()
        }

    def generate_response(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Generate a response using the model
        """
        # Prepare prompt with context if available
        prompt = self.prepare_prompt(query, context)
        
        # Generate response using model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=500,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=1
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def prepare_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Prepare the prompt with context
        """
        if context:
            prompt = f"Context: {json.dumps(context)}\nQuery: {query}\nResponse:"
        else:
            prompt = f"Query: {query}\nResponse:"
        return prompt

    def update_knowledge(self, query: str, response: str):
        """
        Update the knowledge base with new information
        """
        self.knowledge_base[query] = {
            'response': response,
            'timestamp': time.time(),
            'usage_count': self.knowledge_base.get(query, {}).get('usage_count', 0) + 1
        }

    def save_state(self, filepath: str):
        """
        Save AI state to file
        """
        state = {
            'knowledge_base': self.knowledge_base,
            'learning_history': self.learning_history
        }
        with open(filepath, 'w') as f:
            json.dump(state, f)

    def load_state(self, filepath: str):
        """
        Load AI state from file
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
            self.knowledge_base = state['knowledge_base']
            self.learning_history = state['learning_history']

def main():
    # Initialize AI
    ai = CustomAI()
    
    # Example queries with context
    test_queries = [
        {
            'query': "Generate a Python function to calculate fibonacci numbers",
            'context': {'language': 'python', 'difficulty': 'intermediate'}
        },
        {
            'query': "Explain how to implement a binary search tree",
            'context': {'format': 'explanation', 'level': 'beginner'}
        }
    ]
    
    # Process queries
    for test in test_queries:
        print(f"\nUser Query: {test['query']}")
        print(f"Context: {test['context']}")
        
        result = ai.process_query(test['query'], test['context'])
        print(f"\nAI Response: {result['response']}")
        print("-" * 50)
        time.sleep(1)
    
    # Save AI state
    ai.save_state('ai_state.json')

if __name__ == "__main__":
    main()