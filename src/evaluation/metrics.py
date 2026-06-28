from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from src.evaluation.predictor import get_predictions


def compute_metrics(probs, labels, threshold=0.5):
    preds    = (probs >= threshold).astype(int)
    acc      = accuracy_score(labels, preds)
    prec     = precision_score(labels, preds, zero_division=0)
    rec      = recall_score(labels, preds, zero_division=0)
    f1       = f1_score(labels, preds, zero_division=0)
    cm       = confusion_matrix(labels, preds)
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc  = auc(fpr, tpr)

    return {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "confusion_matrix": cm,
        "fpr":              fpr,
        "tpr":              tpr,
        "roc_auc":          roc_auc,
        "preds":            preds
    }


def eval_on_loader(model, dataloader, device):
    probs, labels = get_predictions(model, dataloader, device)
    return compute_metrics(probs, labels)
