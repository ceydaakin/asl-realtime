"""Export a trained checkpoint to Core ML and TFLite and check it still agrees with PyTorch.

    python -m asl_realtime.export --checkpoint runs/transformer_seq_aug/model.pt --out exports/transformer

The exported model takes one window: xy (1, 64, 66, 2) raw landmark x/y and
mask (1, 64) with 1.0 for real frames, and returns class probabilities
(1, 250). The mask is a float because Core ML has no boolean inputs.

Each format comes in two sizes: Core ML with float16 or 8-bit weights, TFLite
with float32 or 8-bit weights (dynamic range). Needs the `export` extra.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import load_split
from .features import USE_DIMS, frame_mask
from .labels import SIGNS
from .landmarks import INPUT_SIZE, N_COLS
from .train import load_model

FORMATS = ("coreml", "tflite")
INPUT_SHAPES = {"xy": (1, INPUT_SIZE, N_COLS, USE_DIMS), "mask": (1, INPUT_SIZE)}


class ExportModel(nn.Module):
    """Float mask in, probabilities out."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model(xy, mask > 0.5).softmax(dim=-1)


def example_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(INPUT_SHAPES["xy"], generator=g)
    mask = torch.ones(INPUT_SHAPES["mask"])
    mask[:, INPUT_SIZE // 2:] = 0.0
    return xy * mask[:, :, None, None], mask


def to_coreml(model: ExportModel, out: Path) -> dict[str, Path]:
    import coremltools as ct

    traced = torch.jit.trace(model, example_inputs())
    fp16 = ct.convert(
        traced,
        inputs=[ct.TensorType(name, shape) for name, shape in INPUT_SHAPES.items()],
        outputs=[ct.TensorType("probs")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )
    fp16.user_defined_metadata["signs"] = json.dumps(SIGNS)
    int8 = ct.optimize.coreml.linear_quantize_weights(
        fp16, ct.optimize.coreml.OptimizationConfig(
            global_config=ct.optimize.coreml.OpLinearQuantizerConfig(mode="linear_symmetric")))
    paths = {"coreml_fp16": out / "asl_fp16.mlpackage", "coreml_int8": out / "asl_int8.mlpackage"}
    fp16.save(str(paths["coreml_fp16"]))
    int8.save(str(paths["coreml_int8"]))
    return paths


def to_tflite(model: ExportModel, out: Path) -> dict[str, Path]:
    import litert_torch
    from ai_edge_quantizer import quantizer, recipe

    paths = {"tflite_fp32": out / "asl_fp32.tflite", "tflite_int8": out / "asl_int8.tflite"}
    litert_torch.convert(model, example_inputs()).export(str(paths["tflite_fp32"]))
    q = quantizer.Quantizer(str(paths["tflite_fp32"]))
    q.load_quantization_recipe(recipe.dynamic_wi8_afp32())
    q.quantize().export_model(str(paths["tflite_int8"]), overwrite=True)
    return paths


def coreml_predictor(path: Path):
    import coremltools as ct

    ml = ct.models.MLModel(str(path))
    return lambda xy, mask: ml.predict({"xy": xy, "mask": mask})["probs"]


def tflite_predictor(path: Path):
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    by_rank = {len(d["shape"]): d["index"] for d in interp.get_input_details()}
    out_index = interp.get_output_details()[0]["index"]

    def predict(xy, mask):
        interp.set_tensor(by_rank[4], xy)
        interp.set_tensor(by_rank[2], mask)
        interp.invoke()
        return interp.get_tensor(out_index)

    return predict


EXPORTERS = {"coreml": (to_coreml, coreml_predictor), "tflite": (to_tflite, tflite_predictor)}


def size_mb(path: Path) -> float:
    files = [path] if path.is_file() else [f for f in path.rglob("*") if f.is_file()]
    return sum(f.stat().st_size for f in files) / 1e6


def check(predict, reference: ExportModel, xy: np.ndarray, mask: np.ndarray, y: np.ndarray) -> dict:
    """Accuracy of an exported model on real clips, and how often it picks the same sign as PyTorch."""
    correct = ref_correct = same = 0
    max_diff, millis = 0.0, []
    for i in range(len(y)):
        start = time.perf_counter()
        probs = np.asarray(predict(xy[i:i + 1], mask[i:i + 1]))
        millis.append((time.perf_counter() - start) * 1000)
        with torch.no_grad():
            ref = reference(torch.from_numpy(xy[i:i + 1]), torch.from_numpy(mask[i:i + 1])).numpy()
        correct += int(probs.argmax() == y[i])
        ref_correct += int(ref.argmax() == y[i])
        same += int(probs.argmax() == ref.argmax())
        max_diff = max(max_diff, float(np.abs(probs - ref).max()))
    n = len(y)
    return {"top1": correct / n, "torch_top1": ref_correct / n, "agreement": same / n, "max_prob_diff": max_diff,
            "host_ms_median": float(np.median(millis))}


def export(checkpoint: Path, out: Path, formats=FORMATS, data_root: str | None = None, limit: int = 2000) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model = ExportModel(load_model(checkpoint)).eval()
    if data_root:
        val = load_split(data_root, "val")
        xy = np.ascontiguousarray(val.X[:limit, ..., :USE_DIMS], dtype=np.float32)
        mask = frame_mask(val.frame_idxs[:limit]).astype(np.float32)
        y = np.asarray(val.y[:limit])
    else:
        ex_xy, ex_mask = example_inputs()
        xy, mask = ex_xy.numpy(), ex_mask.numpy()
        y = model(ex_xy, ex_mask).argmax(dim=-1).numpy()

    report = {"checkpoint": str(checkpoint), "inputs": INPUT_SHAPES, "output": "probs (1, 250)",
              "checked_on": f"{len(y)} val clips" if data_root else "1 synthetic clip", "variants": {}}
    for fmt in formats:
        convert, predictor = EXPORTERS[fmt]
        for name, path in convert(model, out).items():
            report["variants"][name] = {"file": path.name, "size_mb": round(size_mb(path), 3),
                                        **check(predictor(path), model, xy, mask, y)}
    (out / "signs.json").write_text(json.dumps(SIGNS))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--formats", nargs="+", choices=FORMATS, default=list(FORMATS))
    p.add_argument("--data-root", default="data/gislr_public", help="val clips for the accuracy check")
    p.add_argument("--limit", type=int, default=2000)
    a = p.parse_args()
    report = export(a.checkpoint, a.out, a.formats, a.data_root, a.limit)
    for name, v in report["variants"].items():
        print(f"{name:12s} {v['size_mb']:6.2f} MB  top1={v['top1']:.4f} (torch {v['torch_top1']:.4f})  "
              f"same prediction as torch={v['agreement']:.4f}  "
              f"{v['host_ms_median']:.2f} ms (this machine)")


if __name__ == "__main__":
    main()
