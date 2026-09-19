"""Explicit denominators and fixed-label confusion matrices, without point adjustment."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def detection(y, scores, threshold):
    truth = np.asarray(y) > 0
    pred = np.asarray(scores) > threshold
    negative = ~truth
    return {
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "false_positive_rate": float(pred[negative].mean()) if negative.any() else None,
        "pr_auc_ap": float(average_precision_score(truth, scores)) if truth.any() else None,
        "positives": int(truth.sum()),
        "negatives": int(negative.sum()),
        "predicted_anomalies": int(pred.sum()),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(truth, pred, labels=[False, True]).tolist(),
    }


def diagnosis(y, prediction, labels):
    cm = confusion_matrix(y, prediction, labels=labels)
    support = cm.sum(1)
    recalls = [float(cm[i, i] / n) if n else None for i, n in enumerate(support)]
    return {
        "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        "labels": list(labels),
        "recall_per_class": recalls,
        "support_per_class": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def cascade(scores, threshold, anomalous_prediction):
    """Every rejected sample becomes normal, including false negatives."""
    return np.where(np.asarray(scores) > threshold, anomalous_prediction, 0)
