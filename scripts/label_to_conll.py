
# scripts/label_to_conll.py

import pandas as pd
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_FILE = os.path.join(BASE_DIR, "data", "raw", "telegram_data.csv")
LABELED_FILE = os.path.join(BASE_DIR, "data", "labeled", "ner_dataset.conll")

# Create labeled folder if it doesn't exist
os.makedirs(os.path.dirname(LABELED_FILE), exist_ok=True)

# Load raw message data
df = pd.read_csv(RAW_FILE)
df = df.dropna(subset=["text"])

# Sample 50 messages
sampled = df["text"].sample(50, random_state=42).tolist()

# Define simple pattern-based rule function for labeling
def label_tokens(message):
    tokens = message.split()
    labeled = []

    for token in tokens:
        if "ብር" in token or token.isdigit():
            label = "B-PRICE" if not labeled or labeled[-1][1] != "B-PRICE" else "I-PRICE"
        elif any(loc in token for loc in ["አዲስ", "ቦሌ", "ጅማ", "ሃዋሳ", "መስከረም", "ቀን"]):
            label = "B-LOC" if not labeled or labeled[-1][1] != "B-LOC" else "I-LOC"
        elif any(prod in token for prod in ["መቀመጫ", "ጫማ", "ቀሚስ", "ሱሪ", "ሻሚዝ", "አልባስ"]):
            label = "B-Product" if not labeled or labeled[-1][1] != "B-Product" else "I-Product"
        else:
            label = "O"
        labeled.append((token, label))

    return labeled

# Save labeled data to CoNLL format
with open(LABELED_FILE, "w", encoding="utf-8") as f:
    for message in sampled:
        for token, label in label_tokens(message):
            f.write(f"{token}\t{label}\n")
        f.write("\n")  # Blank line to separate messages

print(f"✅ Labeled data saved to: {LABELED_FILE}")
