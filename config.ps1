# Configuration file for Mak AI system

# System configuration
$config = @{
    # General settings
    "debug_mode" = $false
    "log_level" = "INFO"
    
    # Model settings
    "default_model" = "mistralai/Mistral-7B-v0.1"
    "model_cache_dir" = ".model_cache"
    "max_sequence_length" = 2048
    
    # Training settings
    "batch_size" = 4
    "learning_rate" = 1e-5
    "num_epochs" = 3
    "warmup_steps" = 500
    "gradient_accumulation_steps" = 4
    
    # Dataset settings
    "target_datasets" = @(
        "OpenCodeReasoning",
        "LLaVA",
        "Nemotron",
        "CodeAlpaca",
        "StackExchange"
    )
    
    # Wasabi storage settings
    "wasabi_endpoint" = "https://s3.wasabisys.com"
    "wasabi_bucket" = "mak-storage"
    "wasabi_access_key" = ""
    "wasabi_secret_key" = ""
    
    # Telegram bot settings
    "telegram_token" = ""
    "telegram_chat_id" = ""
    
    # Scraping settings
    "scraping_interval" = 3600  # in seconds
    "max_scraped_pages" = 100
    
    # Dashboard settings
    "dashboard_port" = 8501
    "dashboard_host" = "localhost"
    
    # Optimization settings
    "optimization_interval" = 3600  # in seconds
    "max_prompt_variations" = 3
    
    # Memory settings
    "max_memory_usage" = 0.8  # 80% of available memory
    "check_memory_interval" = 300  # in seconds
}

# Save configuration to file
$config | ConvertTo-Json | Set-Content "config.json"

Write-Host "Configuration loaded successfully"
