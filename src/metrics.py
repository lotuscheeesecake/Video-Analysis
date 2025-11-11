# src/metrics.py
import numpy as np
from sklearn.metrics import average_precision_score

def precision_at_k_batch(probabilities, targets, k=5):
    """
    probabilities: (B, L) probabilities
    targets: (B, L) 0/1
    return average precision@k over batch
    """
    B, L = probabilities.shape
    precisions = []
    topk = min(k, L)
    for i in range(B):
        preds = np.argsort(-probabilities[i])[:topk]
        true = np.where(targets[i] == 1)[0]
        if len(true) == 0:
            # ignore or treat as zero precision
            precisions.append(0.0)
        else:
            precisions.append(len(set(preds) & set(true)) / topk)
    return float(np.mean(precisions))

def mean_average_precision(probs, targets):
    """
    Compute mAP (per-class) using sklearn average_precision_score
    probs, targets: np arrays shape (N, L)
    """
    L = targets.shape[1]
    aps = []
    for j in range(L):
        y_true = targets[:, j]
        y_score = probs[:, j]
        # if class is never positive, skip
        if y_true.sum() == 0:
            continue
        try:
            ap = average_precision_score(y_true, y_score)
            aps.append(ap)
        except ValueError:
            pass
    if len(aps) == 0:
        return 0.0
    return float(np.mean(aps))
