import torch

from asl_realtime.augment import AugConfig, augment, augment_time
from asl_realtime.landmarks import HAND_SLICE, INPUT_SIZE, LIPS_SLICE, N_COLS, POSE_SLICE

IDENTITY = AugConfig(scale=(1.0, 1.0), aspect=(1.0, 1.0), rotate_deg=0.0, shear=0.0,
                     drop_lips_p=0.0, drop_pose_p=0.0, drop_point_p=0.0,
                     time_scale=(1.0, 1.0), drop_frame_p=0.0)


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


def _time_cfg(time_scale=(1.0, 1.0), drop_frame_p=0.0):
    return AugConfig(time_scale=time_scale, drop_frame_p=drop_frame_p)


def test_time_identity_config_leaves_input_unchanged():
    xy, mask = _xy()

    out, out_mask = augment_time(xy, mask, IDENTITY, _gen())

    torch.testing.assert_close(out, xy)
    assert torch.equal(out_mask, mask)


def test_time_scale_changes_length_and_keeps_frame_order():
    xy, mask = _xy(batch=2)
    real = INPUT_SIZE - 12
    xy[:, :real, 0, 0] = torch.arange(1, real + 1, dtype=xy.dtype)  # frame id in one coordinate

    slow, slow_mask = augment_time(xy, mask, _time_cfg(time_scale=(1.2, 1.2)), _gen())
    fast, fast_mask = augment_time(xy, mask, _time_cfg(time_scale=(0.5, 0.5)), _gen())

    assert slow_mask.sum(dim=1).tolist() == [round(real * 1.2)] * 2
    assert fast_mask.sum(dim=1).tolist() == [real // 2] * 2
    for out, out_mask in ((slow, slow_mask), (fast, fast_mask)):
        ids = out[0, out_mask[0], 0, 0]
        assert torch.all(ids.diff() >= 0)
        assert ids[0] <= 2 and ids[-1] >= real - 1


def test_frame_dropout_packs_remaining_frames_at_the_start():
    xy, mask = _xy()
    real = INPUT_SIZE - 12
    xy[:, :real, 0, 0] = torch.arange(1, real + 1, dtype=xy.dtype)

    out, out_mask = augment_time(xy, mask, _time_cfg(drop_frame_p=0.5), _gen())

    kept = out_mask.sum(dim=1)
    assert torch.all((kept >= 1) & (kept < real))
    for b in range(len(xy)):
        assert torch.all(out_mask[b, :kept[b]]) and not out_mask[b, kept[b]:].any()
        assert torch.all(out[b, kept[b]:] == 0)
        assert torch.all(out[b, :kept[b], 0, 0].diff() > 0)
        assert torch.all(out[b, :, 7] == 0)


def test_time_augment_never_empties_a_sequence_and_keeps_empty_ones_empty():
    xy, mask = _xy(batch=3)
    mask[0, 1:] = False
    xy[0, 1:] = 0.0
    mask[1] = False
    xy[1] = 0.0

    out, out_mask = augment_time(xy, mask, _time_cfg(time_scale=(0.1, 0.1), drop_frame_p=0.99), _gen())

    assert out_mask.sum(dim=1).tolist()[:2] == [1, 0]
    torch.testing.assert_close(out[0, 0], xy[0, 0])
    assert out_mask[2].sum() >= 1


def test_time_augment_same_seed_gives_same_result():
    xy, mask = _xy()

    a, am = augment_time(xy, mask, AugConfig(), _gen(5))
    b, bm = augment_time(xy, mask, AugConfig(), _gen(5))

    torch.testing.assert_close(a, b)
    assert torch.equal(am, bm)
