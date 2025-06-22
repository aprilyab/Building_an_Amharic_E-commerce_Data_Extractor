import pandas as pd

def label_tokens(msg):
    tokens = msg.split()
    labeled = []
    for token in tokens:
        if "ብር" in token:
            labeled.append((token, "B-PRICE"))
        elif "Addis" in token or "Bole" in token:
            labeled.append((token, "B-LOC"))
        elif "መቀመጫ" in token:
            labeled.append((token, "B-Product"))
        else:
            labeled.append((token, "O"))
    return labeled

def save_conll(messages, filepath="data/labeled/ner_dataset.conll"):
    with open(filepath, "w", encoding="utf-8") as f:
        for msg in messages:
            for token, label in label_tokens(msg):
                f.write(f"{token}\t{label}\n")
            f.write("\n")

df = pd.read_csv("data/raw/telegram_data.csv")
messages = df["text"].dropna().sample(50).tolist()
save_conll(messages)
