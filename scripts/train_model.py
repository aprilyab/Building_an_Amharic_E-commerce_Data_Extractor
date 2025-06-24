# scripts/train_model.py

from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)
from datasets import load_dataset, DatasetDict
from seqeval.metrics import classification_report
import os

model_name = "xlm-roberta-base"  # Change this per model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=label_count)

# Load and preprocess your dataset
train_dataset = ... # Load and tokenize train.conll
val_dataset = ...   # Load and tokenize val.conll

args = TrainingArguments(
    output_dir=f"./models/{model_name.replace('/', '-')}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./results/model_logs",
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

trainer.train()
trainer.evaluate()
trainer.save_model(f"./models/{model_name.replace('/', '-')}")
