import numpy as np
import pytest

from asl_realtime.landmarks import (HAND_SLICE, INPUT_SIZE, LEFT_HAND_IDXS0, LIPS_IDXS0, LIPS_SLICE, N_COLS,
                                    POSE_SLICE, RIGHT_HAND_IDXS0, RIGHT_POSE_IDXS0)
from asl_realtime.live import SignSegmenter, to_window


def _frames(n, hand="left", seed=0):
    """n Holistic frames with face and pose always detected and one hand in view."""
    frames = np.random.default_rng(seed).random((n, 543, 3), dtype=np.float32) * 0.8 + 0.1
    frames[:, LEFT_HAND_IDXS0 if hand != "left" else RIGHT_HAND_IDXS0] = np.nan
    if hand is None:
        frames[:, LEFT_HAND_IDXS0] = np.nan
        frames[:, RIGHT_HAND_IDXS0] = np.nan
    return frames


def test_window_has_model_input_shape_and_keeps_left_dominant_coordinates():
    frames = _frames(10)

    xy, mask = to_window(frames)

    assert xy.shape == (INPUT_SIZE, N_COLS, 2) and xy.dtype == np.float32
    assert mask.tolist() == [True] * 10 + [False] * (INPUT_SIZE - 10)
    np.testing.assert_array_equal(xy[:10, LIPS_SLICE], frames[:, LIPS_IDXS0, :2])
    np.testing.assert_array_equal(xy[:10, HAND_SLICE], frames[:, LEFT_HAND_IDXS0, :2])
    assert np.all(xy[10:] == 0)


def test_right_dominant_clip_is_mirrored_except_for_the_lips():
    frames = _frames(6, hand="right")

    xy, _ = to_window(frames)

    np.testing.assert_array_equal(xy[:6, LIPS_SLICE], frames[:, LIPS_IDXS0, :2])
    np.testing.assert_allclose(xy[:6, HAND_SLICE, 0], 1 - frames[:, RIGHT_HAND_IDXS0, 0], rtol=1e-6)
    np.testing.assert_array_equal(xy[:6, HAND_SLICE, 1], frames[:, RIGHT_HAND_IDXS0, 1])
    np.testing.assert_allclose(xy[:6, POSE_SLICE, 0], 1 - frames[:, RIGHT_POSE_IDXS0, 0], rtol=1e-6)


def test_frames_without_the_dominant_hand_are_dropped_and_missing_points_become_zero():
    frames = _frames(8)
    frames[2:5, LEFT_HAND_IDXS0] = np.nan
    frames[:, LIPS_IDXS0[3]] = np.nan

    xy, mask = to_window(frames)

    assert mask.sum() == 5
    np.testing.assert_array_equal(xy[2, HAND_SLICE], frames[5, LEFT_HAND_IDXS0, :2])
    assert np.all(xy[:, 3] == 0)


def test_long_clip_is_averaged_down_to_the_window_size():
    frames = _frames(INPUT_SIZE * 2)
    frames[:, LEFT_HAND_IDXS0[0], 0] = np.arange(INPUT_SIZE * 2)

    xy, mask = to_window(frames)

    assert mask.all()
    np.testing.assert_allclose(xy[:, HAND_SLICE.start, 0], np.arange(INPUT_SIZE) * 2 + 0.5)


def test_clip_without_hands_gives_an_empty_window():
    xy, mask = to_window(_frames(5, hand=None))

    assert not mask.any() and np.all(xy == 0)


def test_window_rejects_wrong_landmark_count():
    with pytest.raises(ValueError, match="543"):
        to_window(np.zeros((4, 66, 2)))


def test_segmenter_emits_a_sign_after_the_hand_leaves():
    seg = SignSegmenter(gap=3, min_frames=2)
    sign, idle = _frames(5), _frames(6, hand=None)

    assert all(seg.push(f) is None for f in idle[:2])   # nothing started yet
    assert all(seg.push(f) is None for f in sign)
    assert seg.push(idle[0]) is None and seg.push(idle[1]) is None
    out = seg.push(idle[2])

    assert out.shape == (8, 543, 3)
    assert to_window(out)[1].sum() == 5
    assert seg.push(idle[3]) is None


def test_segmenter_ignores_blips_and_cuts_very_long_signs():
    seg = SignSegmenter(gap=2, min_frames=3, max_frames=6)
    blip, idle = _frames(1), _frames(2, hand=None)

    assert [seg.push(f) for f in (*blip, *idle)] == [None, None, None]

    outs = [seg.push(f) for f in _frames(6)]
    assert outs[:5] == [None] * 5 and outs[5].shape[0] == 6
