# scripts/train_ner_model.py

import sys
import os

# Debug info for Python environment
print("Python executable:", sys.executable)

# Add project root to sys.path (adjust if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers
print("Transformers imported from:", transformers.__file__)
print("Transformers version:", transformers.__version__)

from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

# Diagnostic prints to check where TrainingArguments is imported from
print("TrainingArguments location:", TrainingArguments.__module__)
print("TrainingArguments class file:", getattr(TrainingArguments, '__file__', 'N/A'))

from seqeval.metrics import precision_score, recall_score, f1_score

from scripts.config import *
from scripts.read_conll import read_conll
from scripts.prepare_dataset import prepare_dataset

# Step 1: Load data
sentences, labels = read_conll(data_path)

# Step 2: Tokenize and prepare dataset
tokenized_dataset, tokenizer, label2id, id2label = prepare_dataset(sentences, labels)

# Step 3: Load model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save model at the end of each epoch
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_dir=log_dir,
)

print("TrainingArguments created successfully.")

# Step 5: Define metrics computation function
def compute_metrics(p):
    preds = p.predictions.argmax(axis=2)
    true_labels = [
        [id2label[l] for l in label if l != -100] for label in p.label_ids
    ]
    true_preds = [
        [id2label[pred] for pred, lab in zip(pred_row, label) if lab != -100]
        for pred_row, label in zip(preds, p.label_ids)
    ]
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }

# Step 6: Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

print("Trainer created successfully.")

# Step 7: Train the model
trainer.train()

# Step 8: Save the model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Training complete. Model saved in: {output_dir}")
