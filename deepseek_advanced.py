import re
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeBase:
    def __init__(self):
        self.memories: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        
    def add_memory(self, text: str, category: str, timestamp: datetime):
        """Add new knowledge with metadata."""
        memory = {
            "text": text,
            "category": category,
            "timestamp": timestamp,
            "access_count": 0
        }
        self.memories.append(memory)
        self._update_vectors()
    
    def _update_vectors(self):
        """Update vector representations of memories."""
        if self.memories:
            texts = [m["text"] for m in self.memories]
            self.vectors = self.vectorizer.fit_transform(texts)
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search memories by relevance."""
        if not self.memories:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            self.memories[idx]["access_count"] += 1
            results.append(self.memories[idx])
        
        return results

class SelfLearningModule:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.learning_threshold = 0.7
        
    def learn_from_interaction(self, user_input: str, ai_response: str, success_score: float):
        """Learn from successful interactions."""
        if success_score > self.learning_threshold:
            self.knowledge_base.add_memory(
                f"Input: {user_input}\nResponse: {ai_response}",
                "interaction",
                datetime.now()
            )
    
    def learn_from_correction(self, original_response: str, corrected_response: str):
        """Learn from corrections."""
        self.knowledge_base.add_memory(
            f"Correction - Original: {original_response}\nCorrected: {corrected_response}",
            "correction",
            datetime.now()
        )

class ResponseGenerator:
    def __init__(self):
        # Initialize DeepSeek model
        self.model_name = "deepseek-ai/deepseek-coder-6.7b-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            
    def generate_response(self, 
                         prompt: str, 
                         context: List[Dict[str, Any]], 
                         max_length: int = 500) -> str:
        """Generate response using DeepSeek model with context."""
        # Prepare context
        context_text = "\n".join([
            f"Previous {m['category']}: {m['text']}" 
            for m in context[-3:] # Last 3 relevant memories
        ])
        
        # Combine context and prompt
        full_prompt = f"""Context:
{context_text}

Current Input:
{prompt}

Response:"""

        # Generate response
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Response:")[-1].strip()

class DeepSeekAI:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.learner = SelfLearningModule(self.knowledge_base)
        self.response_generator = ResponseGenerator()
        self.conversation_history: List[str] = []
        
    def process_input(self, user_input: str) -> str:
        """Process user input and generate response."""
        # Add to conversation history
        self.conversation_history.append(user_input)
        if len(self.conversation_history) > 10:
            self.conversation_history.pop(0)
            
        # Search relevant knowledge
        relevant_memories = self.knowledge_base.search(user_input)
        
        # Generate response
        response = self.response_generator.generate_response(
            user_input,
            relevant_memories
        )
        
        # Learn from interaction (assuming positive feedback)
        self.learner.learn_from_interaction(
            user_input,
            response,
            success_score=0.8  # This could be based on user feedback
        )
        
        return response
        
    def handle_feedback(self, original_response: str, corrected_response: str):
        """Handle user corrections for learning."""
        self.learner.learn_from_correction(original_response, corrected_response)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "total_memories": len(self.knowledge_base.memories),
            "conversation_turns": len(self.conversation_history),
            "memory_categories": {
                category: len([m for m in self.knowledge_base.memories 
                             if m["category"] == category])
                for category in set(m["category"] for m in self.knowledge_base.memories)
            }
        }

def main():
    ai = DeepSeekAI()
    print("DeepSeek AI initialized. Type 'exit' to quit, 'stats' for system statistics.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'stats':
            print("\nSystem Statistics:")
            print(ai.get_stats())
            continue
            
        try:
            response = ai.process_input(user_input)
            print(f"\nAI: {response}")
            
            # Optional: Get feedback
            feedback = input("\nWas this response helpful? (y/n/correct): ").lower()
            if feedback == 'correct':
                correction = input("Please provide the correct response: ")
                ai.handle_feedback(response, correction)
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different input.")

if __name__ == "__main__":
    main() 