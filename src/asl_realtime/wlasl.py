"""Cross-dataset check: run a model trained on GISLR on WLASL videos of the same signs.

    curl -LO https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json
    python -m asl_realtime.wlasl build --index WLASL_v0.3.json --out data/wlasl
    python -m asl_realtime.wlasl eval --clips data/wlasl/clips.npz --checkpoint runs/transformer_seq_aug/model.pt

WLASL is a list of links to videos on other sites. YouTube and Flash links are
skipped and many of the rest are dead, so the clip count is whatever could
still be downloaded. Needs the `demo` extra.
"""

import argparse
import json
import urllib.request
from multiprocessing import Pool
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

from .labels import SIGNS
from .live import to_window
from .metrics import topk_correct

_MIN_VIDEO_BYTES = 20_000  # anything smaller is an error page, not a video


def overlapping_instances(index: list[dict], signs=SIGNS) -> list[dict]:
    """WLASL instances whose gloss is one of our signs and whose video is a direct download."""
    label = {sign: i for i, sign in enumerate(signs)}
    return [
        {"video_id": inst["video_id"], "url": inst["url"], "label": label[entry["gloss"]],
         "frame_start": inst["frame_start"], "frame_end": inst["frame_end"]}
        for entry in index if entry["gloss"] in label
        for inst in entry["instances"]
        if "youtu" not in urlparse(inst["url"]).netloc and not inst["url"].endswith(".swf")
    ]


def clip_frames(frames: np.ndarray, frame_start: int, frame_end: int) -> np.ndarray:
    """WLASL frame ranges are 1-based and inclusive; frame_end -1 means the end of the video."""
    return frames[frame_start - 1:None if frame_end == -1 else frame_end]


def _download(url: str, path: Path) -> bool:
    if path.exists():
        return True
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(request, timeout=30).read()
    except OSError:
        return False
    if len(data) < _MIN_VIDEO_BYTES:
        return False
    path.write_bytes(data)
    return True


def _extract(job: tuple[dict, str, str]) -> dict | None:
    """Download one video and turn it into a model window. None if the link is dead or no hand is found."""
    from .demo import landmark_stream

    inst, videos, landmarker = job
    path = Path(videos) / f"{inst['video_id']}.mp4"
    if not _download(inst["url"], path):
        return None
    try:
        frames = [frame for _, frame in landmark_stream(str(path), landmarker)]
    except RuntimeError:
        return None
    if not frames:
        return None
    xy, mask = to_window(clip_frames(np.stack(frames), inst["frame_start"], inst["frame_end"]))
    return {"xy": xy, "mask": mask, "y": inst["label"], "video_id": inst["video_id"]} if mask.any() else None


def build(index_path: Path, out: Path, landmarker: str, workers: int = 6) -> dict:
    videos = out / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    instances = overlapping_instances(json.loads(index_path.read_text()))
    with Pool(workers) as pool:
        clips = [c for c in pool.imap_unordered(_extract, [(i, str(videos), landmarker) for i in instances]) if c]
    clips = sorted(clips, key=lambda c: c["video_id"])
    np.savez_compressed(out / "clips.npz", xy=np.stack([c["xy"] for c in clips]),
                        mask=np.stack([c["mask"] for c in clips]), y=np.array([c["y"] for c in clips]),
                        video_id=np.array([c["video_id"] for c in clips]))
    return {"linked": len(instances), "usable": len(clips), "signs": len({c["y"] for c in clips})}


@torch.no_grad()
def evaluate(model, clips_path: Path) -> dict:
    clips = np.load(clips_path)
    logits = model(torch.from_numpy(clips["xy"]), torch.from_numpy(clips["mask"]))
    y = torch.from_numpy(clips["y"])
    hits = topk_correct(logits, y, ks=(1, 5))
    return {"clips": len(y), "signs": int(y.unique().numel()), "top1": hits[1] / len(y), "top5": hits[5] / len(y)}


def main() -> None:
    from .train import load_model

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--index", type=Path, required=True)
    b.add_argument("--out", type=Path, default=Path("data/wlasl"))
    b.add_argument("--landmarker", default="holistic_landmarker.task")
    b.add_argument("--workers", type=int, default=6)
    e = sub.add_parser("eval")
    e.add_argument("--clips", type=Path, default=Path("data/wlasl/clips.npz"))
    e.add_argument("--checkpoint", required=True)
    a = p.parse_args()
    if a.command == "build":
        print(json.dumps(build(a.index, a.out, a.landmarker, a.workers)))
    else:
        print(json.dumps(evaluate(load_model(a.checkpoint), a.clips)))


if __name__ == "__main__":
    main()
