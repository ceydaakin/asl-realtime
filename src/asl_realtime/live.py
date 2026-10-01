"""Turn MediaPipe Holistic frames into the window the model was trained on.

This repeats the preprocessing of the training data (see landmarks.py): keep
only frames where the dominant hand is visible, pick the 66 landmarks, mirror
right-dominant clips, and fit the clip into 64 frames. Clips longer than 64
hand frames are averaged down in equal chunks, which is close to what the
dataset preprocessing does but not identical to it at the edges.

Frames are (543, 2+) arrays in Holistic order with NaN for landmarks that
were not detected.
"""

import warnings

import numpy as np

from .features import USE_DIMS
from .landmarks import (HAND_SLICE, INPUT_SIZE, LANDMARK_IDXS_LEFT_DOMINANT, LANDMARK_IDXS_RIGHT_DOMINANT,
                        LEFT_HAND_IDXS0, N_COLS, N_HOLISTIC_LANDMARKS, RIGHT_HAND_IDXS0)


def _detected(frames: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """(T,) True for frames where any of the given landmarks was detected."""
    return ~np.isnan(frames[:, idxs]).all(axis=(1, 2))


def to_window(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """frames: (T, 543, 2+). Returns xy (64, 66, 2) float32 and mask (64,) bool."""
    frames = np.asarray(frames, np.float32)[..., :USE_DIMS]
    if frames.ndim != 3 or frames.shape[1] != N_HOLISTIC_LANDMARKS:
        raise ValueError(f"Expected shape (T, {N_HOLISTIC_LANDMARKS}, 2+), got {frames.shape}")
    left, right = _detected(frames, LEFT_HAND_IDXS0), _detected(frames, RIGHT_HAND_IDXS0)
    left_dominant = left.sum() >= right.sum()
    clip = frames[left if left_dominant else right]
    if left_dominant:
        clip = clip[:, LANDMARK_IDXS_LEFT_DOMINANT]
    else:
        clip = clip[:, LANDMARK_IDXS_RIGHT_DOMINANT]
        mirrored = clip.copy()
        mirrored[:, HAND_SLICE.start:, 0] = 1.0 - clip[:, HAND_SLICE.start:, 0]  # hand and arm, not the lips
        clip = mirrored
    if len(clip) > INPUT_SIZE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a landmark missing in a whole chunk stays NaN
            clip = np.stack([np.nanmean(chunk, axis=0) for chunk in np.array_split(clip, INPUT_SIZE)])

    xy = np.zeros((INPUT_SIZE, N_COLS, USE_DIMS), np.float32)
    xy[:len(clip)] = np.nan_to_num(clip, nan=0.0)
    return xy, np.arange(INPUT_SIZE) < len(clip)


class SignSegmenter:
    """Cuts a live stream into signs: a sign is a run of frames with a hand in view.

    push() returns the frames of a finished sign once no hand has been seen for
    `gap` frames, or once `max_frames` have piled up, and None otherwise.
    """

    def __init__(self, gap: int = 8, min_frames: int = 4, max_frames: int = 4 * INPUT_SIZE):
        self.gap = gap
        self.min_frames = min_frames
        self.max_frames = max_frames
        self._frames: list[np.ndarray] = []
        self._hand_frames = 0
        self._missing = 0

    def push(self, frame: np.ndarray) -> np.ndarray | None:
        hand = _detected(frame[None], np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0)))[0]
        if not hand and not self._frames:
            return None
        self._frames = [*self._frames, frame]
        self._hand_frames += int(hand)
        self._missing = 0 if hand else self._missing + 1
        if self._missing < self.gap and len(self._frames) < self.max_frames:
            return None
        frames, enough = np.stack(self._frames), self._hand_frames >= self.min_frames
        self._frames, self._hand_frames, self._missing = [], 0, 0
        return frames if enough else None
