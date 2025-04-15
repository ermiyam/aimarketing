import os
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import yaml

class ChatExecutor:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['base_model'],
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['tokenizer'],
            trust_remote_code=True
        )
        
        # Set chat template
        self.tokenizer.chat_template = "chatml"
        
    def execute_command(self, command):
        """Execute a command and return the output"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )
            return result.stdout if result.returncode == 0 else result.stderr
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def chat(self, user_input):
        """Process user input and execute commands if needed"""
        # Check if input is a command (starts with !)
        if user_input.startswith('!'):
            command = user_input[1:].strip()
            return self.execute_command(command)
        
        # Otherwise, generate response using the model
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant that can execute commands and provide information."},
            {"role": "user", "content": user_input}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

def main():
    executor = ChatExecutor()
    print("Chat Executor initialized. Type 'exit' to quit.")
    print("Prefix commands with '!' to execute them (e.g., '!ls')")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        response = executor.chat(user_input)
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main() 