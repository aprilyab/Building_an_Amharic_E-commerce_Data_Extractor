# scripts/scrape_telegram.py
from telethon import TelegramClient
import os
from dotenv import load_dotenv
import pandas as pd
import re

load_dotenv()
api_id = os.getenv("TG_API_ID")
api_hash = os.getenv("TG_API_HASH")
phone = os.getenv("PHONE")

client = TelegramClient('ethio_session', api_id, api_hash)

def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r"[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\u0300-\u036F\s\w.,!?]", "", text)
    return re.sub(r"\s+", " ", text).strip()

async def scrape_and_save(channels):
    all_data = []
    for channel in channels:
        async for message in client.iter_messages(channel, limit=100):
            if message.message:
                all_data.append({
                    "channel": channel,
                    "text": clean_text(message.message),
                    "date": message.date,
                    "views": message.views,
                    "id": message.id
                })

    df = pd.DataFrame(all_data)
    df.to_csv("data/raw/telegram_data.csv", index=False)
    print("Saved to data/raw/telegram_data.csv")

with client:
    channels = ['shageronlinestore', 'ethiobeststore', 'addisstore', 'telemarketeth', 'ethiomartdaily']
    client.loop.run_until_complete(scrape_and_save(channels))
