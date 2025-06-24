# scripts/evaluate_model.py

from seqeval.metrics import f1_score, precision_score, recall_score

# Assume predictions and true labels are available
def evaluate(preds, labels):
    f1 = f1_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    return f1, prec, rec
