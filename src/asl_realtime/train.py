"""Train a landmark classifier on the preprocessed GISLR splits.

    python -m asl_realtime.train --model conv1d --norm sequence --augment --epochs 60
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .augment import AugConfig, augment, augment_time
from .data import GislrSplit, load_split
from .features import N_FEATURES, USE_DIMS, FeatureStats, compute_stats, frame_mask
from .landmarks import INPUT_SIZE, LANDMARK_IDXS_LEFT_DOMINANT, LANDMARK_IDXS_RIGHT_DOMINANT
from .metrics import topk_correct
from .models import MODELS, LandmarkClassifier, build_model
from .normalize import MODES, Normalizer


@dataclass(frozen=True)
class TrainConfig:
    model: str = "gru"
    data_root: str = "data/gislr_public"
    out_dir: str = "runs/gru"
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    norm: str = "sequence"
    augment: bool = False
    aug: AugConfig = field(default_factory=AugConfig)
    device: str = "auto"
    limit: int | None = None
    seed: int = 42


PREPROCESSING = {
    "input_size": INPUT_SIZE,
    "use_dims": USE_DIMS,
    "landmark_idxs_left_dominant": LANDMARK_IDXS_LEFT_DOMINANT.tolist(),
    "landmark_idxs_right_dominant": LANDMARK_IDXS_RIGHT_DOMINANT.tolist(),
    "right_dominant_mirroring": "x -> 1 - x for hand and pose points",
    "missing_value": 0.0,
    "model_input": "raw x/y in [0, 1] image coordinates; normalization is inside the model",
}


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_classifier(name: str, norm: str, stats: FeatureStats | None = None, **net_kwargs) -> LandmarkClassifier:
    net = build_model(name, n_features=N_FEATURES, **net_kwargs)
    return LandmarkClassifier(name, Normalizer(norm, stats if norm == "global" else None), net)


def load_model(path: str | Path, device: str = "cpu") -> LandmarkClassifier:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    hp = dict(ckpt["model_hparams"])
    name, norm = hp.pop("model"), hp.pop("norm")
    stats = FeatureStats.from_dict(ckpt["stats"]) if ckpt.get("stats") else None
    model = build_classifier(name, norm, stats, **hp)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def _to_dataset(split: GislrSplit, limit: int | None) -> TensorDataset:
    n = len(split) if limit is None else min(limit, len(split))
    xy = torch.from_numpy(np.ascontiguousarray(split.X[:n, ..., :USE_DIMS], dtype=np.float32))
    mask = torch.from_numpy(frame_mask(split.frame_idxs[:n]))
    y = torch.from_numpy(np.asarray(split.y[:n], np.int64))
    return TensorDataset(xy, mask, y)


def _param_groups(model, weight_decay: float) -> list[dict]:
    """No weight decay on biases and norm weights (all 1-D params)."""
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def _train_epoch(model, loader, opt, sched, device, config: TrainConfig) -> float:
    model.train()
    total, n = 0.0, 0
    for xy, mask, y in loader:
        xy, mask, y = xy.to(device), mask.to(device), y.to(device)
        if config.augment:
            xy, mask = augment_time(xy, mask, config.aug)
            xy = augment(xy, mask, config.aug)
        loss = F.cross_entropy(model(xy, mask), y, label_smoothing=config.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    loss, hits, n = 0.0, {1: 0, 5: 0}, 0
    for xy, mask, y in loader:
        xy, mask, y = xy.to(device), mask.to(device), y.to(device)
        logits = model(xy, mask)
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        for k, v in topk_correct(logits, y, ks=(1, 5)).items():
            hits[k] += v
        n += len(y)
    return {"val_loss": loss / n, "val_top1": hits[1] / n, "val_top5": hits[5] / n}


def _save(out: Path, model: LandmarkClassifier, config: TrainConfig, stats, metrics: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                "model_hparams": model.hparams, "n_features": N_FEATURES,
                "stats": stats.to_dict() if stats else None, "preprocessing": PREPROCESSING},
               out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))


def _fit(model, train_dl, val_dl, device, config: TrainConfig, on_best) -> list[dict]:
    opt = torch.optim.AdamW(_param_groups(model, config.weight_decay), lr=config.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.lr, total_steps=config.epochs * len(train_dl), pct_start=0.1)
    history, best = [], None
    for epoch in range(1, config.epochs + 1):
        start = time.time()
        train_loss = _train_epoch(model, train_dl, opt, sched, device, config)
        row = {"epoch": epoch, "train_loss": train_loss, **evaluate(model, val_dl, device),
               "seconds": round(time.time() - start, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or row["val_top1"] > best["val_top1"]:
            best = row
            on_best(best, history)
    return history


def run(config: TrainConfig) -> dict:
    if config.norm not in MODES:
        raise ValueError(f"Unknown norm {config.norm!r}, expected one of {MODES}")
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_split = load_split(config.data_root, "train")
    val_split = load_split(config.data_root, "val")
    stats = compute_stats(train_split.X) if config.norm == "global" else None
    train_ds, val_ds = _to_dataset(train_split, config.limit), _to_dataset(val_split, config.limit)
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size * 2)
    model = build_classifier(config.model, config.norm, stats).to(device)

    base = {"model": config.model, "norm": config.norm, "augment": config.augment,
            "params": sum(p.numel() for p in model.parameters()),
            "n_train": len(train_ds), "n_val": len(val_ds), "device": str(device)}

    def on_best(best, history):
        _save(out, model, config, stats, {**base, **best, "epochs": best["epoch"], "history": history})

    history = _fit(model, train_dl, val_dl, device, config, on_best)
    best = max(history, key=lambda r: r["val_top1"])
    # Best epoch is picked on val, so val_top1 is optimistic; last_* is the unbiased number.
    metrics = {**base, **best, "epochs": config.epochs, "best_epoch": best["epoch"],
               "last_val_top1": history[-1]["val_top1"], "last_val_top5": history[-1]["val_top5"],
               "history": history}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(MODELS), default="gru")
    p.add_argument("--norm", choices=MODES, default=TrainConfig.norm)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--data-root", default=TrainConfig.data_root)
    p.add_argument("--out-dir")
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--device", default=TrainConfig.device)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    a = p.parse_args()
    config = TrainConfig(model=a.model, norm=a.norm, augment=a.augment, data_root=a.data_root,
                         out_dir=a.out_dir or f"runs/{a.model}", epochs=a.epochs,
                         batch_size=a.batch_size, lr=a.lr, device=a.device, limit=a.limit, seed=a.seed)
    m = run(config)
    print(f"best val top1={m['val_top1']:.4f} top5={m['val_top5']:.4f} (epoch {m['best_epoch']}), "
          f"last top1={m['last_val_top1']:.4f} top5={m['last_val_top5']:.4f}")


if __name__ == "__main__":
    main()
