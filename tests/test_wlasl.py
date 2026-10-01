import numpy as np
import torch

from asl_realtime.landmarks import INPUT_SIZE, N_COLS
from asl_realtime.wlasl import clip_frames, evaluate, overlapping_instances

INDEX = [
    {"gloss": "book", "instances": [
        {"video_id": "1", "url": "https://example.org/book.mp4", "frame_start": 1, "frame_end": -1},
        {"video_id": "2", "url": "https://www.youtube.com/watch?v=x", "frame_start": 1, "frame_end": -1},
        {"video_id": "3", "url": "https://example.org/book.swf", "frame_start": 1, "frame_end": -1},
    ]},
    {"gloss": "computer", "instances": [
        {"video_id": "4", "url": "https://example.org/computer.mp4", "frame_start": 1, "frame_end": -1},
    ]},
    {"gloss": "dog", "instances": [
        {"video_id": "5", "url": "https://youtu.be/y", "frame_start": 1, "frame_end": -1},
        {"video_id": "6", "url": "https://example.org/many.mp4", "frame_start": 30, "frame_end": 60},
    ]},
]


def test_overlapping_instances_keeps_direct_downloads_of_known_signs():
    out = overlapping_instances(INDEX, signs=("book", "dog"))

    assert [(i["video_id"], i["label"]) for i in out] == [("1", 0), ("6", 1)]
    assert out[1]["frame_start"] == 30 and out[1]["frame_end"] == 60


def test_clip_frames_uses_one_based_inclusive_range():
    frames = np.arange(100)

    assert clip_frames(frames, 1, -1).tolist() == list(range(100))
    assert clip_frames(frames, 30, 60).tolist() == list(range(29, 60))


def test_evaluate_reports_accuracy_over_saved_clips(tmp_path):
    y = np.array([30, 7, 7, 100])
    np.savez_compressed(tmp_path / "clips.npz", xy=np.zeros((4, INPUT_SIZE, N_COLS, 2), np.float32),
                        mask=np.ones((4, INPUT_SIZE), bool), y=y, video_id=np.array(list("abcd")))

    def always_seven(xy, mask):
        logits = torch.zeros(len(xy), 250)
        logits[:, 7] = 1.0
        return logits

    assert evaluate(always_seven, tmp_path / "clips.npz") == {"clips": 4, "signs": 3, "top1": 0.5, "top5": 0.5}
