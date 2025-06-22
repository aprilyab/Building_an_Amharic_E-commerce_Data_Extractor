# scripts/scrape_telegram.py

from telethon import TelegramClient
import os
from dotenv import load_dotenv
import pandas as pd
import re

# Load credentials from .env
load_dotenv()
api_id = os.getenv("TG_API_ID")
api_hash = os.getenv("TG_API_HASH")
phone = os.getenv("PHONE")

# Initialize Telegram client session
client = TelegramClient('ethio_session', api_id, api_hash)

# Define the project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Clean Amharic messages
def clean_text(text):
    if text is None:
        return ""
    # Remove non-Amharic symbols while keeping punctuation
    text = re.sub(r"[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\u0300-\u036F\s\w.,!?]", "", text)
    return re.sub(r"\s+", " ", text).strip()

# Main scraper logic
async def scrape_and_save(channels):
    all_data = []
    for channel in channels:
        try:
            print(f"📥 Scraping: {channel}")
            async for message in client.iter_messages(channel, limit=100):
                if message.message:
                    all_data.append({
                        "channel": channel,
                        "text": clean_text(message.message),
                        "date": message.date,
                        "views": message.views,
                        "id": message.id
                    })
        except Exception as e:
            print(f"[!] Skipping channel '{channel}': {e}")

    # Save to correct data/raw/telegram_data.csv
    RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "telegram_data.csv")
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df = pd.DataFrame(all_data)
    df.to_csv(RAW_PATH, index=False)
    print(f"✅ Saved to {RAW_PATH}")

# Start the session and scrape
async def main():
    await client.start(phone=phone)
    print("🔌 Logged in.")

    # Only the 8 valid Telegram e-commerce channels
    channels = [
        'ZemenExpress',
        'nevacomputer',
        'meneshayeofficial',
        'ethio_brand_collection',
        'Leyueqa',
        'sinayelj',
        'Shewabrand',
        'helloomarketethiopia'
    ]

    await scrape_and_save(channels)

# Run the script
with client:
    client.loop.run_until_complete(main())
