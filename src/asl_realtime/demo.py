"""Recognize signs from a webcam or a video file on this machine. Needs the `demo` extra.

    curl -LO https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
    python -m asl_realtime.demo --checkpoint runs/transformer_seq_aug/model.pt
    python -m asl_realtime.demo --checkpoint runs/transformer_seq_aug/model.pt --video clip.mp4

With a webcam, raise your hand, sign, and lower it: the sign is classified when
the hand leaves the picture. A video file is treated as one sign.
"""

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .labels import SIGNS
from .landmarks import N_HOLISTIC_LANDMARKS
from .live import SignSegmenter, to_window

# Holistic order: face, left hand, pose, right hand.
_PARTS = (("face_landmarks", 0, 468), ("left_hand_landmarks", 468, 21),
          ("pose_landmarks", 489, 33), ("right_hand_landmarks", 522, 21))


def holistic_frame(result) -> np.ndarray:
    """MediaPipe HolisticLandmarkerResult -> (543, 3), NaN for parts that were not detected."""
    frame = np.full((N_HOLISTIC_LANDMARKS, 3), np.nan, np.float32)
    for field, start, count in _PARTS:
        points = getattr(result, field)
        if points:
            frame[start:start + count] = [(p.x, p.y, p.z) for p in points[:count]]
    return frame


def landmark_stream(source: int | str, landmarker: str | Path) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yields (BGR image, holistic frame) for every frame of a camera index or video file."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HolisticLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(landmarker)),
                                               running_mode=vision.RunningMode.VIDEO)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source {source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        with vision.HolisticLandmarker.create_from_options(options) as detector:
            index = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    return
                rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                yield image, holistic_frame(detector.detect_for_video(rgb, int(index * 1000 / fps)))
                index += 1
    finally:
        cap.release()


@torch.no_grad()
def predict(model, frames: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
    """Top-k (sign, probability) for one clip of holistic frames. Empty if no hand was seen."""
    xy, mask = to_window(frames)
    if not mask.any():
        return []
    probs = model(torch.from_numpy(xy)[None], torch.from_numpy(mask)[None]).softmax(dim=-1)[0]
    top = probs.topk(k)
    return [(SIGNS[i], float(p)) for p, i in zip(top.values, top.indices)]


def _format(top: list[tuple[str, float]]) -> str:
    return ", ".join(f"{sign} {p:.0%}" for sign, p in top) or "no hand detected"


def run_video(model, path: str, landmarker: str) -> list[tuple[str, float]]:
    frames = np.stack([frame for _, frame in landmark_stream(path, landmarker)])
    return predict(model, frames)


def run_camera(model, camera: int, landmarker: str) -> None:
    import cv2

    segmenter, caption = SignSegmenter(), "raise your hand and sign"
    for image, frame in landmark_stream(camera, landmarker):
        clip = segmenter.push(frame)
        if clip is not None:
            caption = _format(predict(model, clip, k=3))
            print(caption, flush=True)
        cv2.putText(image, caption, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("asl-realtime (q to quit)", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


def main() -> None:
    from .train import load_model

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--landmarker", default="holistic_landmarker.task")
    p.add_argument("--video", help="video file with one sign; omit to use the webcam")
    p.add_argument("--camera", type=int, default=0)
    a = p.parse_args()
    model = load_model(a.checkpoint)
    if a.video:
        print(_format(run_video(model, a.video, a.landmarker)))
    else:
        run_camera(model, a.camera, a.landmarker)


if __name__ == "__main__":
    main()
