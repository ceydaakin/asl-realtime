"""Loading the preprocessed GISLR splits.

The train/val split is participant-disjoint (GroupShuffleSplit on
participant_id, 10% of participants held out), so val accuracy measures
generalization to signers the model has never seen.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .landmarks import INPUT_SIZE, N_COLS, N_DIMS, NUM_CLASSES

SPLITS = ("train", "val")


@dataclass(frozen=True)
class GislrSplit:
    X: np.ndarray           # (N, INPUT_SIZE, N_COLS, N_DIMS) float32, memory-mapped
    y: np.ndarray           # (N,) int, sign index in [0, NUM_CLASSES)
    frame_idxs: np.ndarray  # (N, INPUT_SIZE) float32, source frame index, -1 for padding

    def __len__(self) -> int:
        return len(self.y)


def _load(path: Path, mmap: bool = False) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name} in {path.parent}")
    return np.load(path, mmap_mode="r" if mmap else None)


def load_split(root: str | Path, split: str) -> GislrSplit:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}, expected one of {SPLITS}")
    root = Path(root)

    X = _load(root / f"X_{split}.npy", mmap=True)
    y = _load(root / f"y_{split}.npy")
    frame_idxs = _load(root / f"NON_EMPTY_FRAME_IDXS_{split.upper()}.npy")

    expected = (INPUT_SIZE, N_COLS, N_DIMS)
    if X.ndim != 4 or X.shape[1:] != expected:
        raise ValueError(f"X_{split} has shape {X.shape}, expected (N, {', '.join(map(str, expected))})")
    if not len(X) == len(y) == len(frame_idxs):
        raise ValueError(
            f"{split} length mismatch: X={len(X)}, y={len(y)}, frame_idxs={len(frame_idxs)}"
        )
    if len(y) and (y.min() < 0 or y.max() >= NUM_CLASSES):
        raise ValueError(f"{split} has label outside [0, {NUM_CLASSES}): {y.min()}..{y.max()}")

    return GislrSplit(X=X, y=y, frame_idxs=frame_idxs)
