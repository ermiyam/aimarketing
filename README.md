# gen

A specialized language model for content generation and analysis.

## Features

- Code generation with specialized models
- Content analysis and optimization
- Performance tracking and metrics
- Real-time response generation
- Attention pattern visualization

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ermiyam/gen.git
cd gen
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the chat interface:
```bash
python src/ai_model/response_handler.py
```

## Project Structure

```
gen/
├── src/
│   ├── ai_model/
│   │   ├── response_handler.py
│   │   └── code_gen_model.py
│   └── api/
│       └── server.py
├── logs/
├── cache/
├── checkpoints/
├── requirements.txt
└── README.md
```

## API Endpoints

- POST `/api/chat`: Send messages and get responses
- GET `/api/stats`: Get performance metrics
- GET `/health`: Check server status

## Development

This project uses:
- Python 3.8+
- PyTorch
- Transformers
- Flask
- Cursor IDE
- GitHub for version control

## Syncing Changes

To sync changes with GitHub, use the PowerShell script:
```powershell
.\sync.ps1
```

## License

MIT License

# Mak ReSearch Training System

A powerful training system for Mak AI that integrates search-based reasoning with GRPO (Generalized Reinforcement Policy Optimization).

## 🚀 Features

- **Search-Integrated Training**: Model learns when and what to search
- **GRPO Loss**: Custom loss function with search result masking
- **Multi-Hop Reasoning**: Support for complex, multi-step reasoning
- **Reflexive Learning**: Self-correction and improvement capabilities
- **Flexible Search**: Combines embedding and keyword-based search
- **Comprehensive Evaluation**: Multiple metrics for tracking progress

## 📋 Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- 20GB+ disk space

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mak-research.git
cd mak-research
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

1. Copy and edit the config template:
```bash
cp config.yaml.example config.yaml
```

2. Update the configuration in `config.yaml`:
```yaml
model:
  base_model: "Qwen/Qwen1.5-7B-Chat"
  tokenizer: "Qwen/Qwen1.5-7B-Chat"
  use_bfloat16: true

training:
  batch_size: 2
  max_seq_len: 2048
  num_epochs: 3
  learning_rate: 2e-5
  gradient_accumulation_steps: 4

search:
  embedding_model: "BAAI/bge-large-en-v1.5"
  keyword_model: "BM25"
  fusion_strategy: "weighted"
  embedding_weight: 0.7
  keyword_weight: 0.3

grpo:
  alpha: 0.1
  beta: 0.5
  gamma: 0.3
  max_kl: 0.1
  clip_range: 0.2
  value_clip_range: 0.2
  entropy_coef: 0.01
```

## 🏃‍♂️ Training

1. Start training:
```bash
python train_mak_research.py
```

2. Monitor training:
```bash
python dashboard.py
```

3. Evaluate model:
```bash
python eval_research.py
```

## 📊 Training Process

1. **Data Loading**
   - Loads MuSiQue dataset
   - Optionally includes custom transcripts
   - Preprocesses data for training

2. **Model Training**
   - Generates search queries
   - Retrieves relevant information
   - Computes GRPO loss
   - Updates model parameters

3. **Evaluation**
   - Computes multiple metrics
   - Tracks search effectiveness
   - Monitors reasoning quality

## 📈 Metrics

- ROUGE scores (1, 2, L)
- BLEU score
- Accuracy
- F1 score
- Search relevance
- Reasoning chain quality

## 🤖 Inference

```python
from prompt_templates import format_inference_prompt
from search_handler import SearchHandler

# Initialize components
search_handler = SearchHandler()
model = AutoModelForCausalLM.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Generate response
def generate_response(question):
    prompt = format_inference_prompt(question)
    response = model.generate(prompt)
    return response
```

## 📚 Dataset Structure

```
data/
├── musique/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── transcripts/
│   └── custom_data.jsonl
└── cache/
    └── search_results.json
```

## 🔧 Troubleshooting

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Search Issues**
   - Check search index
   - Verify API keys
   - Monitor rate limits

3. **Training Instability**
   - Adjust learning rate
   - Modify GRPO parameters
   - Check data quality

## 📝 License

MIT License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For support, email support@mak.ai or open an issue.
