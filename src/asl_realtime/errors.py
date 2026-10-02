"""Per-sign error analysis of a classifier's predictions."""

import numpy as np


def per_sign_accuracy(y: np.ndarray, pred: np.ndarray, signs: list[str]) -> list[dict]:
    """One row per sign that occurs in `y`, least accurate first."""
    rows = [{"sign": signs[c], "clips": int((y == c).sum()), "top1": float((pred[y == c] == c).mean())}
            for c in np.unique(y)]
    return sorted(rows, key=lambda r: (r["top1"], r["sign"]))


def top_confusions(y: np.ndarray, pred: np.ndarray, signs: list[str], n: int = 20) -> list[dict]:
    """The `n` most frequent (true sign, predicted sign) mistakes."""
    wrong = y != pred
    pairs, counts = np.unique(np.stack([y[wrong], pred[wrong]], axis=1), axis=0, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:n]
    return [{"true": signs[t], "predicted": signs[p], "count": int(counts[i]),
             "share_of_true": float(counts[i] / (y == t).sum())}
            for i in order for t, p in [pairs[i]]]
