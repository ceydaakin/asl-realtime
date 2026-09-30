import numpy as np
import pytest

from asl_realtime.data import GislrSplit, load_split
from asl_realtime.landmarks import INPUT_SIZE, N_COLS, N_DIMS, NUM_CLASSES


def _write_split(root, split, n=5, frames=INPUT_SIZE, cols=N_COLS, dims=N_DIMS, labels=None):
    rng = np.random.default_rng(0)
    np.save(root / f"X_{split}.npy", rng.random((n, frames, cols, dims), dtype=np.float32))
    y = np.arange(n) % NUM_CLASSES if labels is None else np.asarray(labels)
    np.save(root / f"y_{split}.npy", y.astype(np.int32))
    idxs = np.tile(np.arange(frames, dtype=np.float32), (n, 1))
    np.save(root / f"NON_EMPTY_FRAME_IDXS_{split.upper()}.npy", idxs)


def test_load_split_returns_arrays_with_expected_shapes(tmp_path):
    _write_split(tmp_path, "train", n=7)

    split = load_split(tmp_path, "train")

    assert isinstance(split, GislrSplit)
    assert split.X.shape == (7, INPUT_SIZE, N_COLS, N_DIMS)
    assert split.y.shape == (7,)
    assert split.frame_idxs.shape == (7, INPUT_SIZE)
    assert len(split) == 7


def test_load_split_memory_maps_landmarks(tmp_path):
    _write_split(tmp_path, "val")

    split = load_split(tmp_path, "val")

    assert isinstance(split.X, np.memmap)


def test_load_split_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="split"):
        load_split(tmp_path, "test")


def test_load_split_reports_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="X_train.npy"):
        load_split(tmp_path, "train")


def test_load_split_rejects_wrong_landmark_layout(tmp_path):
    _write_split(tmp_path, "train", cols=N_COLS + 1)

    with pytest.raises(ValueError, match="shape"):
        load_split(tmp_path, "train")


def test_load_split_rejects_length_mismatch(tmp_path):
    _write_split(tmp_path, "train", n=5)
    np.save(tmp_path / "y_train.npy", np.zeros(4, dtype=np.int32))

    with pytest.raises(ValueError, match="length"):
        load_split(tmp_path, "train")


def test_load_split_rejects_out_of_range_labels(tmp_path):
    _write_split(tmp_path, "train", n=3, labels=[0, 1, NUM_CLASSES])

    with pytest.raises(ValueError, match="label"):
        load_split(tmp_path, "train")
