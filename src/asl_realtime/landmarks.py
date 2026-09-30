"""Landmark layout of the preprocessed GISLR data.

Each MediaPipe Holistic frame has 543 landmarks: 468 face, 21 left hand,
33 pose and 21 right hand. Preprocessing keeps 66 of them per frame:
40 lip points, the 21 points of the dominant hand and 5 points of the
dominant arm. Right-dominant sequences are mirrored on x so every sample
looks left-dominant. The live camera pipeline must apply the same selection
and mirroring, otherwise the model sees inputs it was never trained on.

Source: markwijkhuizen/gislr-tf-data-processing-transformer-training (Kaggle).
"""

import numpy as np

N_HOLISTIC_LANDMARKS = 543
INPUT_SIZE = 64
N_DIMS = 3
NUM_CLASSES = 250

LIPS_IDXS0 = np.array([
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
])
LEFT_HAND_IDXS0 = np.arange(468, 489)
RIGHT_HAND_IDXS0 = np.arange(522, 543)
LEFT_POSE_IDXS0 = np.array([502, 504, 506, 508, 510])
RIGHT_POSE_IDXS0 = np.array([503, 505, 507, 509, 511])

LANDMARK_IDXS_LEFT_DOMINANT = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, LEFT_POSE_IDXS0))
LANDMARK_IDXS_RIGHT_DOMINANT = np.concatenate((LIPS_IDXS0, RIGHT_HAND_IDXS0, RIGHT_POSE_IDXS0))
N_COLS = LANDMARK_IDXS_LEFT_DOMINANT.size

LIPS_SLICE = slice(0, len(LIPS_IDXS0))
HAND_SLICE = slice(LIPS_SLICE.stop, LIPS_SLICE.stop + len(LEFT_HAND_IDXS0))
POSE_SLICE = slice(HAND_SLICE.stop, N_COLS)
