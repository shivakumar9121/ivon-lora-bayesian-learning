"""Multiclass calibration metrics. Probabilities and ECE use the [0, 1] scale."""
import numpy as np


def validate(probs, labels):
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise ValueError("Expected nonempty N by C probabilities, with C >= 2")
    if y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Labels must be a one-dimensional integer array")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0):
        raise ValueError("Every probability row must sum to one")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("Label index is outside the class range")
    return p, y


def reliability_bins(probs, labels, n_bins=15):
    p, y = validate(probs, labels)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    confidence = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    # [lo, hi), except the last bin includes confidence = 1 exactly.
    index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows = []
    for i in range(n_bins):
        mask = index == i
        n = int(mask.sum())
        rows.append({"lower": i / n_bins, "upper": (i + 1) / n_bins,
                     "count": n,
                     "confidence": float(confidence[mask].mean()) if n else None,
                     "accuracy": float(correct[mask].mean()) if n else None})
    return rows


def classification_metrics(probs, labels, n_bins=15):
    p, y = validate(probs, labels)
    bins = reliability_bins(p, y, n_bins)
    ece = sum(b["count"] / len(y) * abs(b["accuracy"] - b["confidence"])
              for b in bins if b["count"])
    # Natural log, clamp only to make exact zero probabilities finite.
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    one_hot = np.eye(p.shape[1])[y]
    return {"accuracy": float((p.argmax(1) == y).mean()), "ece": float(ece),
            "nll": float(nll), "brier": float(np.square(p - one_hot).sum(1).mean()),
            "n": len(y), "ece_bins": n_bins}


def average_probabilities(sample_probs):
    p = np.asarray(sample_probs, dtype=np.float64)
    if p.ndim != 3 or len(p) == 0:
        raise ValueError("Expected samples by examples by classes")
    for sample in p:
        validate(sample, np.zeros(len(sample), dtype=int))
    return p.mean(axis=0)
