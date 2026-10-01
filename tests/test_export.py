import json

import numpy as np
import pytest
import torch

from asl_realtime.export import INPUT_SHAPES, ExportModel, example_inputs, export, size_mb
from asl_realtime.train import TrainConfig, _save, build_classifier


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(0)
    model = build_classifier("transformer", "sequence", hidden=32, layers=1, heads=2, ff=64).eval()
    _save(tmp_path, model, TrainConfig(model="transformer"), None, {})
    return tmp_path / "model.pt"


def test_export_model_takes_float_mask_and_returns_probabilities():
    model = build_classifier("conv1d", "sequence", hidden=32, blocks=1).eval()
    xy, mask = example_inputs()

    with torch.no_grad():
        probs = ExportModel(model)(xy, mask)
        logits = model(xy, mask.bool())

    assert tuple(xy.shape) == INPUT_SHAPES["xy"] and tuple(mask.shape) == INPUT_SHAPES["mask"]
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(probs, logits.softmax(dim=-1))


def test_size_mb_counts_files_and_directories(tmp_path):
    (tmp_path / "pkg" / "inner").mkdir(parents=True)
    (tmp_path / "pkg" / "a.bin").write_bytes(b"x" * 1_000_000)
    (tmp_path / "pkg" / "inner" / "b.bin").write_bytes(b"x" * 500_000)

    assert size_mb(tmp_path / "pkg" / "a.bin") == 1.0
    assert size_mb(tmp_path / "pkg") == 1.5


@pytest.mark.parametrize("fmt, module, files", [
    ("coreml", "coremltools", ["asl_fp16.mlpackage", "asl_int8.mlpackage"]),
    ("tflite", "litert_torch", ["asl_fp32.tflite", "asl_int8.tflite"]),
])
def test_export_writes_models_that_agree_with_pytorch(tmp_path, checkpoint, fmt, module, files):
    pytest.importorskip(module)
    out = tmp_path / "export"

    report = export(checkpoint, out, formats=[fmt])

    assert all((out / f).exists() for f in files)
    assert len(json.loads((out / "signs.json").read_text())) == 250
    assert json.loads((out / "report.json").read_text())["variants"].keys() == report["variants"].keys()
    full, small = (report["variants"][k] for k in report["variants"])
    assert full["agreement"] == 1.0 and full["max_prob_diff"] < 1e-2
    assert small["size_mb"] < full["size_mb"]
    assert np.isfinite(small["max_prob_diff"])
