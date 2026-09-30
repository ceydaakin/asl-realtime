import numpy as np
import pytest

from asl_realtime.features import FeatureStats, compute_stats, frame_mask, normalize
from asl_realtime.landmarks import INPUT_SIZE, N_COLS, N_DIMS

USE_DIMS = 2


def _batch(n=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, INPUT_SIZE, N_COLS, N_DIMS), dtype=np.float32) + 0.1
    X[:, -8:] = 0.0          # padded frames
    X[:, :, :5] = 0.0        # landmarks missing in every frame
    return X


def test_compute_stats_ignores_missing_zeros():
    X = _batch()

    stats = compute_stats(X)

    assert stats.mean.shape == stats.std.shape == (N_COLS, USE_DIMS)
    present = X[:, :-8, 5:, :USE_DIMS].reshape(-1, N_COLS - 5, USE_DIMS)
    np.testing.assert_allclose(stats.mean[5:], present.mean(axis=0), rtol=1e-5)
    assert np.all(stats.std > 0)


def test_compute_stats_handles_landmark_never_present():
    X = _batch()

    stats = compute_stats(X)

    np.testing.assert_array_equal(stats.mean[:5], 0.0)
    np.testing.assert_array_equal(stats.std[:5], 1.0)


def test_compute_stats_resets_only_the_degenerate_dim():
    X = _batch()
    X[..., 10, 1] = 0.5  # y of landmark 10 is constant, x still varies

    stats = compute_stats(X)

    assert stats.std[10, 1] == 1.0
    assert stats.std[10, 0] != 1.0


def test_normalize_keeps_missing_points_at_zero_and_drops_z():
    X = _batch()
    stats = compute_stats(X)

    out = normalize(X, stats)

    assert out.shape == (len(X), INPUT_SIZE, N_COLS * USE_DIMS)
    assert out.dtype == np.float32
    reshaped = out.reshape(len(X), INPUT_SIZE, N_COLS, USE_DIMS)
    assert np.all(reshaped[:, -8:] == 0)
    assert np.all(reshaped[:, :, :5] == 0)


def test_normalize_standardizes_present_points():
    X = _batch(n=64)
    stats = compute_stats(X)

    out = normalize(X, stats).reshape(len(X), INPUT_SIZE, N_COLS, USE_DIMS)

    present = out[:, :-8, 5:]
    assert abs(present.mean()) < 0.05
    assert abs(present.std() - 1) < 0.05


def test_frame_mask_marks_padding():
    idxs = np.array([[0, 1, 2, -1, -1], [0, 3, 7, 9, 12]], dtype=np.float32)

    mask = frame_mask(idxs)

    np.testing.assert_array_equal(mask, [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    assert mask.dtype == np.bool_


def test_stats_round_trip_through_dict():
    stats = compute_stats(_batch())

    restored = FeatureStats.from_dict(stats.to_dict())

    np.testing.assert_array_equal(restored.mean, stats.mean)
    np.testing.assert_array_equal(restored.std, stats.std)


def test_compute_stats_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        compute_stats(np.zeros((2, INPUT_SIZE, N_COLS + 1, N_DIMS), dtype=np.float32))
