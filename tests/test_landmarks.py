import numpy as np

from asl_realtime.landmarks import (
    HAND_SLICE,
    LANDMARK_IDXS_LEFT_DOMINANT,
    LANDMARK_IDXS_RIGHT_DOMINANT,
    LIPS_SLICE,
    N_COLS,
    POSE_SLICE,
)


def test_layout_is_lips_then_hand_then_pose():
    assert (LIPS_SLICE.start, LIPS_SLICE.stop) == (0, 40)
    assert (HAND_SLICE.start, HAND_SLICE.stop) == (40, 61)
    assert (POSE_SLICE.start, POSE_SLICE.stop) == (61, 66)
    assert N_COLS == 66


def test_dominant_layouts_share_lips_and_differ_in_hand_and_pose():
    left, right = LANDMARK_IDXS_LEFT_DOMINANT, LANDMARK_IDXS_RIGHT_DOMINANT

    assert len(left) == len(right) == N_COLS
    np.testing.assert_array_equal(left[LIPS_SLICE], right[LIPS_SLICE])
    assert set(left[HAND_SLICE]).isdisjoint(right[HAND_SLICE])
    assert set(left[POSE_SLICE]).isdisjoint(right[POSE_SLICE])


def test_indices_point_into_mediapipe_holistic_frame():
    # MediaPipe Holistic frame: 468 face + 21 left hand + 33 pose + 21 right hand = 543
    for idxs in (LANDMARK_IDXS_LEFT_DOMINANT, LANDMARK_IDXS_RIGHT_DOMINANT):
        assert idxs.min() >= 0
        assert idxs.max() < 543
