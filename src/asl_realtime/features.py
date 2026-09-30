"""Turning landmark arrays into model inputs.

Missing landmarks and padded frames are stored as exact zeros. They must stay
zero after normalization and must not leak into the statistics, otherwise a
hand that is out of frame would look like a hand at the image corner.
"""

from dataclasses import dataclass

import numpy as np

from .landmarks import INPUT_SIZE, N_COLS, N_DIMS

USE_DIMS = 2  # x, y. MediaPipe's z estimate is too noisy to help.
N_FEATURES = N_COLS * USE_DIMS
_CHUNK = 4096


@dataclass(frozen=True)
class FeatureStats:
    mean: np.ndarray  # (N_COLS, USE_DIMS)
    std: np.ndarray   # (N_COLS, USE_DIMS)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureStats":
        return cls(mean=np.asarray(d["mean"], np.float32), std=np.asarray(d["std"], np.float32))


def _check_shape(X: np.ndarray) -> None:
    if X.ndim != 4 or X.shape[1:] != (INPUT_SIZE, N_COLS, N_DIMS):
        raise ValueError(f"Expected shape (N, {INPUT_SIZE}, {N_COLS}, {N_DIMS}), got {X.shape}")


def _present(xy: np.ndarray) -> np.ndarray:
    """True where a landmark was detected, per point (…, N_COLS, 1)."""
    return np.any(xy != 0, axis=-1, keepdims=True)


def compute_stats(X: np.ndarray) -> FeatureStats:
    """Per-landmark mean/std over detected points only. Works on memmaps in chunks."""
    _check_shape(X)
    total = np.zeros((N_COLS, USE_DIMS), np.float64)
    total_sq = np.zeros_like(total)
    count = np.zeros((N_COLS, 1), np.float64)
    for start in range(0, len(X), _CHUNK):
        xy = np.asarray(X[start:start + _CHUNK, ..., :USE_DIMS], np.float64)
        present = _present(xy)
        total += np.where(present, xy, 0).sum(axis=(0, 1))
        total_sq += np.where(present, xy ** 2, 0).sum(axis=(0, 1))
        count += present.sum(axis=(0, 1))

    safe = np.maximum(count, 1)
    mean = total / safe
    std = np.sqrt(np.maximum(total_sq / safe - mean ** 2, 0))
    never_seen = count[:, 0] == 0
    mean[never_seen] = 0.0
    std[never_seen] = 1.0
    std[std < 1e-6] = 1.0
    return FeatureStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def normalize(X: np.ndarray, stats: FeatureStats) -> np.ndarray:
    """(N, INPUT_SIZE, N_COLS, N_DIMS) -> (N, INPUT_SIZE, N_FEATURES), missing points stay 0."""
    _check_shape(X)
    out = np.empty((len(X), INPUT_SIZE, N_FEATURES), np.float32)
    for start in range(0, len(X), _CHUNK):
        xy = np.asarray(X[start:start + _CHUNK, ..., :USE_DIMS], np.float32)
        scaled = np.where(_present(xy), (xy - stats.mean) / stats.std, 0.0)
        out[start:start + len(xy)] = scaled.reshape(len(xy), INPUT_SIZE, N_FEATURES)
    return out


def frame_mask(frame_idxs: np.ndarray) -> np.ndarray:
    """True for real frames, False for padding (stored as -1)."""
    return frame_idxs != -1
