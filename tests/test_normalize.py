import numpy as np
import pytest
import torch

from asl_realtime.features import USE_DIMS, compute_stats
from asl_realtime.landmarks import INPUT_SIZE, N_COLS, N_DIMS
from asl_realtime.normalize import Normalizer


def _xy(batch=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(batch, INPUT_SIZE, N_COLS, USE_DIMS, generator=g) * 0.6 + 0.2
    mask = torch.ones(batch, INPUT_SIZE, dtype=torch.bool)
    mask[:, -10:] = False
    xy[:, -10:] = 0.0      # padded frames
    xy[:, :, 3:6] = 0.0    # landmarks never detected
    return xy, mask


def _present(xy):
    return (xy != 0).any(-1, keepdim=True)


def test_output_is_flat_features_with_missing_points_zero():
    xy, mask = _xy()

    out = Normalizer("sequence")(xy, mask)

    assert out.shape == (3, INPUT_SIZE, N_COLS * USE_DIMS)
    grid = out.view(3, INPUT_SIZE, N_COLS, USE_DIMS)
    assert torch.all(grid[:, -10:] == 0)
    assert torch.all(grid[:, :, 3:6] == 0)


def test_sequence_mode_ignores_position_and_distance_to_camera():
    xy, mask = _xy()
    p = _present(xy)
    moved = torch.where(p, (xy - 0.5) * 0.6 + torch.tensor([0.62, 0.41]), xy)

    norm = Normalizer("sequence")

    torch.testing.assert_close(norm(moved, mask), norm(xy, mask), atol=1e-5, rtol=1e-4)


def test_sequence_mode_standardizes_each_sample():
    xy, mask = _xy()

    grid = Normalizer("sequence")(xy, mask).view(3, INPUT_SIZE, N_COLS, USE_DIMS)

    real = grid[:, :-10][:, :, [c for c in range(N_COLS) if c not in (3, 4, 5)]]
    for sample in real:
        assert abs(sample.mean().item()) < 1e-4
        assert abs(sample.std().item() - 1) < 0.01


def test_sequence_mode_handles_sample_with_nothing_detected():
    xy, mask = _xy()
    xy[1] = 0.0

    out = Normalizer("sequence")(xy, mask)

    assert torch.isfinite(out).all()
    assert torch.all(out[1] == 0)


def test_global_mode_uses_training_stats():
    rng = np.random.default_rng(0)
    X = rng.random((8, INPUT_SIZE, N_COLS, N_DIMS), dtype=np.float32) + 0.1
    stats = compute_stats(X)
    xy = torch.from_numpy(X[..., :USE_DIMS])
    mask = torch.ones(8, INPUT_SIZE, dtype=torch.bool)

    out = Normalizer("global", stats).eval()(xy, mask).view(8, INPUT_SIZE, N_COLS, USE_DIMS)

    expected = (xy - torch.from_numpy(stats.mean)) / torch.from_numpy(stats.std)
    torch.testing.assert_close(out, expected)


def test_global_mode_requires_stats():
    with pytest.raises(ValueError, match="stats"):
        Normalizer("global")


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        Normalizer("minmax")


def test_stats_are_saved_in_state_dict():
    stats = compute_stats(np.random.default_rng(0).random((4, INPUT_SIZE, N_COLS, N_DIMS), dtype=np.float32))

    state = Normalizer("global", stats).state_dict()

    assert set(state) == {"mean", "std"}
