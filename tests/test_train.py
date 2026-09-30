import json

import numpy as np
import torch

from asl_realtime.landmarks import INPUT_SIZE, N_COLS, N_DIMS
from asl_realtime.train import TrainConfig, run


def _write_split(root, split, n):
    rng = np.random.default_rng(1)
    np.save(root / f"X_{split}.npy", rng.random((n, INPUT_SIZE, N_COLS, N_DIMS), dtype=np.float32))
    np.save(root / f"y_{split}.npy", (np.arange(n) % 4).astype(np.int32))
    np.save(root / f"NON_EMPTY_FRAME_IDXS_{split.upper()}.npy",
            np.tile(np.arange(INPUT_SIZE, dtype=np.float32), (n, 1)))


def test_run_trains_and_writes_metrics_and_checkpoint(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_split(data, "train", 32)
    _write_split(data, "val", 8)
    out = tmp_path / "run"
    config = TrainConfig(model="gru", data_root=str(data), out_dir=str(out),
                         epochs=2, batch_size=8, device="cpu")

    metrics = run(config)

    assert set(metrics) >= {"val_top1", "val_top5", "epochs", "params"}
    assert 0 <= metrics["val_top1"] <= metrics["val_top5"] <= 1
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["val_top1"] == metrics["val_top1"]
    assert len(saved["history"]) == 2

    assert {"last_val_top1", "last_val_top5"} <= set(metrics)

    ckpt = torch.load(out / "model.pt", weights_only=False)
    assert ckpt["config"]["model"] == "gru"
    assert {"mean", "std"} <= set(ckpt["stats"])
    assert "hidden" in ckpt["model_hparams"]
    assert ckpt["preprocessing"]["use_dims"] == 2
    assert len(ckpt["preprocessing"]["landmark_idxs_left_dominant"]) == 66


def test_run_respects_limit(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_split(data, "train", 40)
    _write_split(data, "val", 20)
    config = TrainConfig(model="conv1d", data_root=str(data), out_dir=str(tmp_path / "run"),
                         epochs=1, batch_size=8, device="cpu", limit=10)

    metrics = run(config)

    assert metrics["n_train"] == 10
    assert metrics["n_val"] == 10
