import os
import logging
import asyncio
from datetime import datetime
import json
from telegram_bot import TelegramBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/startup.log'),
        logging.StreamHandler()
    ]
)

# Load configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config: {str(e)}")
        return {}

# Create initial training data
initial_training = [
    "### Input:\nHow to write a viral Instagram caption?\n### Response:\nStart with a hook that creates curiosity, add emotional triggers, and end with a clear CTA. Use power words and make it personal.",
    "### Input:\nWhat makes a YouTube video go viral?\n### Response:\nHook viewers in the first 3 seconds, maintain fast pacing, tell a story, and deliver value quickly. End with a strong call to action.",
    "### Input:\nHow to write TikTok captions that convert?\n### Response:\nKeep it short, use trending sounds, show before/after, and add urgency. Use emojis and hashtags strategically.",
    "### Input:\nWhat's the best way to start a marketing video?\n### Response:\nOpen with a problem your audience faces, show social proof, and promise a solution. Make it emotional and relatable.",
    "### Input:\nHow to write email subject lines that get opened?\n### Response:\nUse personalization, create urgency, and add curiosity. Keep it short and test different variations.",
    "### Input:\nWhat makes a landing page convert?\n### Response:\nClear value proposition, social proof, urgency, and a strong CTA. Remove distractions and focus on one action.",
    "### Input:\nHow to write product descriptions that sell?\n### Response:\nFocus on benefits not features, use sensory words, add social proof, and create urgency. Make it scannable.",
    "### Input:\nWhat's the best way to structure a sales page?\n### Response:\nHook > Problem > Solution > Proof > Offer > CTA. Use subheadings, bullet points, and testimonials.",
    "### Input:\nHow to write Facebook ads that convert?\n### Response:\nTarget specific pain points, use eye-catching visuals, add social proof, and test different angles. Keep copy short.",
    "### Input:\nWhat makes a good marketing hook?\n### Response:\nCreate curiosity, show a problem, promise a solution, or share a surprising fact. Make it emotional and relatable."
]

async def main():
    # Load configuration
    config = load_config()
    if not config.get('telegram_token') or not config.get('telegram_chat_id'):
        logging.warning("Telegram bot token or chat ID not configured. Alerts will not be sent.")
    
    # Initialize Telegram bot
    bot = TelegramBot()
    
    try:
        # Create necessary directories
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        # Save initial training data
        with open("data/train.txt", "w", encoding="utf-8") as f:
            for example in initial_training:
                f.write(example + "\n")

        logging.info("✅ Initial training data created with 10 high-quality examples")
        
        # Send startup notification
        if config.get('telegram_token') and config.get('telegram_chat_id'):
            await bot.send_status_update("🚀 Starting Mak's learning process...")

        # Start the learning process
        print("\n🚀 Starting Mak's learning process...")
        print("1. Initial training data loaded")
        print("2. Model will begin training")
        print("3. Check logs/startup.log for progress")

        # Run the training
        import subprocess
        result = subprocess.run(["python", "src/learn.py"], capture_output=True, text=True)

        if result.returncode == 0:
            logging.info("✅ Initial training completed successfully")
            if config.get('telegram_token') and config.get('telegram_chat_id'):
                await bot.send_status_update("🎉 Mak is ready to learn more!")
            
            print("\n🎉 Mak is ready to learn more!")
            print("Next steps:")
            print("1. Run 'python src/conversation.py' to chat with Mak")
            print("2. Add more training examples to data/train.txt")
            print("3. Run 'python src/learn.py' to train on new data")
        else:
            error_msg = f"Training failed: {result.stderr}"
            logging.error(error_msg)
            if config.get('telegram_token') and config.get('telegram_chat_id'):
                await bot.send_error_alert(error_msg)
            print("\n❌ Training failed. Check logs/startup.log for details.")
            
    except Exception as e:
        error_msg = f"Error in learning process: {str(e)}"
        logging.error(error_msg)
        if config.get('telegram_token') and config.get('telegram_chat_id'):
            await bot.send_error_alert(error_msg)
        print("\n❌ An error occurred. Check logs/startup.log for details.")

if __name__ == "__main__":
    asyncio.run(main()) 