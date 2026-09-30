import torch

from asl_realtime.augment import AugConfig, augment
from asl_realtime.landmarks import HAND_SLICE, INPUT_SIZE, LIPS_SLICE, N_COLS, POSE_SLICE

IDENTITY = AugConfig(scale=(1.0, 1.0), aspect=(1.0, 1.0), rotate_deg=0.0, shear=0.0,
                     drop_lips_p=0.0, drop_pose_p=0.0, drop_point_p=0.0)


def _xy(batch=4):
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(batch, INPUT_SIZE, N_COLS, 2, generator=g) * 0.6 + 0.2
    mask = torch.ones(batch, INPUT_SIZE, dtype=torch.bool)
    mask[:, -12:] = False
    xy[:, -12:] = 0.0
    xy[:, :, 7] = 0.0
    return xy, mask


def _gen(seed=1):
    return torch.Generator().manual_seed(seed)


def test_identity_config_leaves_input_unchanged():
    xy, mask = _xy()

    out = augment(xy, mask, IDENTITY, _gen())

    torch.testing.assert_close(out, xy)


def test_default_config_changes_detected_points_only():
    xy, mask = _xy()

    out = augment(xy, mask, AugConfig(), _gen())

    assert out.shape == xy.shape
    assert not torch.allclose(out[:, :-12], xy[:, :-12])
    assert torch.all(out[:, -12:] == 0)
    assert torch.all(out[:, :, 7] == 0)


def test_same_seed_gives_same_result():
    xy, mask = _xy()

    a = augment(xy, mask, AugConfig(), _gen(5))
    b = augment(xy, mask, AugConfig(), _gen(5))

    torch.testing.assert_close(a, b)


def test_affine_is_applied_around_image_center():
    xy, mask = _xy(batch=1)
    xy[0, 0, 0] = torch.tensor([0.5, 0.5])
    cfg = AugConfig(scale=(1.5, 1.5), aspect=(1.0, 1.0), rotate_deg=30.0, shear=0.0,
                    drop_lips_p=0.0, drop_pose_p=0.0, drop_point_p=0.0)

    out = augment(xy, mask, cfg, _gen())

    torch.testing.assert_close(out[0, 0, 0], torch.tensor([0.5, 0.5]))


def test_group_dropout_removes_whole_group_and_never_the_hand():
    xy, mask = _xy()
    cfg = AugConfig(scale=(1.0, 1.0), aspect=(1.0, 1.0), rotate_deg=0.0, shear=0.0,
                    drop_lips_p=1.0, drop_pose_p=1.0, drop_point_p=0.0)

    out = augment(xy, mask, cfg, _gen())

    assert torch.all(out[:, :, LIPS_SLICE] == 0)
    assert torch.all(out[:, :, POSE_SLICE] == 0)
    torch.testing.assert_close(out[:, :, HAND_SLICE], xy[:, :, HAND_SLICE])
