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
    assert "hidden" in ckpt["model_hparams"]
    assert ckpt["preprocessing"]["use_dims"] == 2
    assert ckpt["model_hparams"]["norm"] == "sequence"
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


def test_run_with_global_norm_and_augmentation(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    _write_split(data, "train", 16)
    _write_split(data, "val", 8)
    config = TrainConfig(model="conv1d", data_root=str(data), out_dir=str(tmp_path / "run"),
                         epochs=1, batch_size=8, device="cpu", norm="global", augment=True)

    metrics = run(config)

    ckpt = torch.load(tmp_path / "run" / "model.pt", weights_only=False)
    assert ckpt["model_hparams"]["norm"] == "global"
    assert "normalizer.mean" in ckpt["state_dict"]
    assert {"mean", "std"} <= set(ckpt["stats"])
    assert metrics["augment"] is True


def test_load_model_restores_classifier_from_checkpoint(tmp_path):
    from asl_realtime.train import load_model

    data = tmp_path / "data"
    data.mkdir()
    _write_split(data, "train", 16)
    _write_split(data, "val", 8)
    out = tmp_path / "run"
    run(TrainConfig(model="gru", data_root=str(data), out_dir=str(out), epochs=1, batch_size=8, device="cpu"))

    model = load_model(out / "model.pt")
    xy = torch.rand(2, INPUT_SIZE, N_COLS, 2)
    mask = torch.ones(2, INPUT_SIZE, dtype=torch.bool)

    assert model(xy, mask).shape == (2, 250)


def test_run_rejects_unknown_norm(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="norm"):
        run(TrainConfig(norm="minmax", out_dir=str(tmp_path / "run")))
