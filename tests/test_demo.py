from types import SimpleNamespace

import numpy as np
import torch

from asl_realtime.demo import holistic_frame, predict
from asl_realtime.landmarks import LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0


def _points(n, value):
    return [SimpleNamespace(x=value, y=value + 0.1, z=0.0) for _ in range(n)]


def test_holistic_frame_places_parts_in_holistic_order_and_marks_missing_ones():
    result = SimpleNamespace(face_landmarks=_points(478, 0.1), pose_landmarks=_points(33, 0.3),
                             left_hand_landmarks=[], right_hand_landmarks=_points(21, 0.4))

    frame = holistic_frame(result)

    assert frame.shape == (543, 3)
    np.testing.assert_allclose(frame[[0, 467], 0], 0.1)
    assert np.isnan(frame[LEFT_HAND_IDXS0]).all()
    np.testing.assert_allclose(frame[[489, 521], 0], 0.3)
    np.testing.assert_allclose(frame[RIGHT_HAND_IDXS0, :2], [[0.4, 0.5]] * 21)


def test_predict_returns_top_signs_and_nothing_without_a_hand():
    def model(xy, mask):
        logits = torch.zeros(1, 250)
        logits[0, 0], logits[0, 249] = 3.0, 2.0
        return logits

    frames = np.random.default_rng(0).random((6, 543, 3), dtype=np.float32)
    no_hands = frames.copy()
    no_hands[:, LEFT_HAND_IDXS0] = np.nan
    no_hands[:, RIGHT_HAND_IDXS0] = np.nan

    top = predict(model, frames, k=2)

    assert [sign for sign, _ in top] == ["TV", "zipper"]
    assert top[0][1] > top[1][1]
    assert predict(model, no_hands) == []
