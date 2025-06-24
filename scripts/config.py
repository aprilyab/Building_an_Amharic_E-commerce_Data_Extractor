# scripts/config.py

model_checkpoint = "xlm-roberta-base"

# Paths
data_path = "data/labeled/ner_dataset.conll"
output_dir = "models/amharic-ner-model"
log_dir = "logs/"

# Training
num_epochs = 5
batch_size = 8
learning_rate = 2e-5
