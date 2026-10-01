"""app/web/preprocess.js is a port of live.py. Both are checked against the same fixture.

Regenerate after changing the preprocessing on purpose:
    python tests/test_web_fixture.py
"""

import json
from pathlib import Path

import numpy as np

from asl_realtime.landmarks import LEFT_HAND_IDXS0, LIPS_IDXS0, RIGHT_HAND_IDXS0
from asl_realtime.live import to_window

FIXTURE = Path(__file__).resolve().parents[1] / "app" / "web" / "fixtures" / "window.json"


def _cases() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    right = rng.random((5, 543, 2)).round(4).astype(np.float32)
    right[:, LEFT_HAND_IDXS0] = np.nan
    right[1, RIGHT_HAND_IDXS0] = np.nan     # hand lost for one frame
    right[:, LIPS_IDXS0[:4]] = np.nan       # some lip points never detected
    left = rng.random((4, 543, 2)).round(4).astype(np.float32)
    left[:3, RIGHT_HAND_IDXS0] = np.nan     # both hands in the last frame, left wins
    return {"right_dominant": right, "left_dominant": left}


def build() -> list[dict]:
    out = []
    for name, frames in _cases().items():
        xy, mask = to_window(frames)
        out.append({"name": name, "frames": np.where(np.isnan(frames), None, frames.astype(np.float64).round(4).astype(object)).tolist(),
                    "xy": xy.round(6).tolist(), "mask": mask.astype(int).tolist()})
    return out


def test_fixture_matches_python_preprocessing():
    for case, fresh in zip(json.loads(FIXTURE.read_text()), build(), strict=True):
        assert case["name"] == fresh["name"]
        assert case["frames"] == fresh["frames"]
        assert case["mask"] == fresh["mask"]
        np.testing.assert_allclose(case["xy"], fresh["xy"], atol=1e-6)


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(build(), separators=(",", ":")))
    print(f"Wrote {FIXTURE}")
