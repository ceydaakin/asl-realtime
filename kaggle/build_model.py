"""Stage the trained transformer as a Kaggle model with one variation per format.

    python kaggle/build_model.py --owner <kaggle-username> [--public]
    kaggle models create -p kaggle/model
    for d in kaggle/model/variations/*; do kaggle models variations create -p "$d"; done
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SLUG = "asl-realtime-transformer"
TITLE = "ASL Realtime Transformer"
SUBTITLE = "Isolated ASL sign recognition from MediaPipe landmarks, 250 signs, 1.3 M parameters"
LICENSE = "Attribution 4.0 International (CC BY 4.0)"
REPO = "https://github.com/ceydaakin/asl-realtime"
TRAINING_DATA = ["markwijkhuizen/gislr-dataset-public"]
SIGNS_FILE = "signs.json"


@dataclass(frozen=True)
class Variation:
    slug: str
    framework: str
    source: str  # file or .mlpackage directory, relative to the exports dir (or the checkpoint)
    report_key: str | None
    format_notes: str
    usage: str


VARIATIONS = [
    Variation(
        "default", "pyTorch", "model.pt", None,
        "PyTorch checkpoint: `state_dict`, model hyperparameters and training config. "
        f"Load it with `asl_realtime.train.load_model` from [the project repo]({REPO}).",
        "```python\nfrom asl_realtime.train import load_model\n\n"
        'model = load_model("model.pt")              # eval mode, CPU\n'
        "probs = model(xy, mask.bool()).softmax(-1)   # xy (B, 64, 66, 2), mask (B, 64)\n```\n\n"
        "Unlike the exports, the checkpoint takes a boolean mask and returns logits.",
    ),
    Variation(
        "fp32", "tfLite", "asl_fp32.tflite", "tflite_fp32",
        "TFLite flatbuffer with float32 weights.",
        "```python\nfrom ai_edge_litert.interpreter import Interpreter\n\n"
        'interp = Interpreter("asl_fp32.tflite")\ninterp.allocate_tensors()\n```',
    ),
    Variation(
        "int8", "tfLite", "asl_int8.tflite", "tflite_int8",
        "TFLite flatbuffer with 8-bit quantized weights. This is the file the browser demo runs.",
        "```python\nfrom ai_edge_litert.interpreter import Interpreter\n\n"
        'interp = Interpreter("asl_int8.tflite")\ninterp.allocate_tensors()\n```',
    ),
    Variation(
        "coreml-fp16", "other", "asl_fp16.mlpackage", "coreml_fp16",
        "Core ML `.mlpackage` (zipped) with float16 weights, for iOS and macOS.",
        '```python\nimport coremltools as ct\n\nmodel = ct.models.MLModel("asl_fp16.mlpackage")\n```',
    ),
    Variation(
        "coreml-int8", "other", "asl_int8.mlpackage", "coreml_int8",
        "Core ML `.mlpackage` (zipped) with 8-bit palettized weights, for iOS and macOS.",
        '```python\nimport coremltools as ct\n\nmodel = ct.models.MLModel("asl_int8.mlpackage")\n```',
    ),
]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def model_card(metrics: dict, report: dict) -> str:
    rows = "\n".join(
        f"| `{v.framework}/{v.slug}` | {r['size_mb']:.2f} MB | {_pct(r['top1'])} | {_pct(r['agreement'])} |"
        for v in VARIATIONS if v.report_key and (r := report["variants"][v.report_key])
    )
    return f"""# Model Summary

A temporal transformer that recognizes 250 isolated American Sign Language signs from
MediaPipe Holistic landmarks instead of pixels. It is small enough to run on a phone or in a
browser, so no video has to leave the device. Code, training loop, export and demos:
[{REPO}]({REPO}).

# Model Characteristics

- Architecture: {metrics['model']} encoder over landmark sequences, {metrics['params']:,} parameters
- Input: 64 frames x 66 landmarks x (x, y), plus a 64-frame mask marking real frames.
  The 66 landmarks are 40 lip points, 21 points of the dominant hand and 5 arm points.
  Right-dominant clips are mirrored so every clip looks left-dominant.
- Normalization: each clip is centered and scaled inside the model ({metrics['norm']} normalization)
- Output: probabilities over 250 signs; `{SIGNS_FILE}` maps the index to the sign name
- Trained from scratch for {metrics['epochs']} epochs with affine, landmark-dropout and temporal augmentation

# Data Overview

Trained on the Google Isolated Sign Language Recognition data (Kaggle competition `asl-signs`),
using the preprocessed public dataset `{TRAINING_DATA[0]}`: {metrics['n_train']:,} training clips
and {metrics['n_val']:,} validation clips. The validation clips come from **participants the model
never saw in training**.

# Evaluation Results

| Split | Top-1 | Top-5 |
|---|---|---|
| Held-out signers ({metrics['n_val']:,} clips) | {_pct(metrics['val_top1'])} | {_pct(metrics['val_top5'])} |
| WLASL, 831 clips of the 200 shared signs, no fine-tuning | 55.1% | 75.3% |

Chance is 0.4%. Three training seeds give 74.6 +/- 0.2% top-1 on held-out signers.

Exported formats, checked on all validation clips ("Same as PyTorch" is how often the export
picks the same sign as the checkpoint):

| Variation | Size | Top-1 | Same as PyTorch |
|---|---|---|---|
{rows}

# Limitations

Isolated signs only, not continuous signing or fingerspelling. Accuracy drops about 20 points
on WLASL: different signers, studio framing, regional sign variants and two-handed signs.
Latency on real phones has not been measured. Not suitable for interpreting or any use where
a wrong sign has consequences.
"""


def instance_usage(v: Variation) -> str:
    return f"""# Model Format

{v.format_notes}

# Training Data

Google Isolated Sign Language Recognition landmarks (`{TRAINING_DATA[0]}`), signer-disjoint split.

# Model Inputs

- `xy`: float32 `(1, 64, 66, 2)`, MediaPipe x/y coordinates of 40 lip, 21 dominant-hand and 5 arm landmarks
- `mask`: float32 `(1, 64)`, 1 for real frames and 0 for padding

# Model Outputs

Probabilities over 250 signs, shape `(1, 250)`. `{SIGNS_FILE}` lists the sign names in index order.

# Model Usage

{v.usage}

Turning camera frames into `xy` and `mask` is done by `asl_realtime/live.py` in [the repo]({REPO}).

# Fine-tuning

Only the PyTorch variation is meant for fine-tuning; the others are inference exports.

# Changelog

- v1: first release
"""


def _copy(src: Path, dst_dir: Path) -> None:
    if src.is_dir():  # .mlpackage is a directory; Kaggle stores files
        shutil.make_archive(str(dst_dir / src.name), "zip", root_dir=src.parent, base_dir=src.name)
    else:
        shutil.copy2(src, dst_dir / src.name)


def build(out_dir: Path, owner: str, exports: Path, run: Path, public: bool = False) -> Path:
    metrics = json.loads((run / "metrics.json").read_text())
    report = json.loads((exports / "report.json").read_text())
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "model-metadata.json").write_text(json.dumps({
        "ownerSlug": owner, "title": TITLE, "slug": MODEL_SLUG, "subtitle": SUBTITLE,
        "isPrivate": not public, "description": model_card(metrics, report),
        "publishTime": "", "provenanceSources": REPO,
    }, indent=2))
    for v in VARIATIONS:
        vdir = out_dir / "variations" / f"{v.framework}-{v.slug}"
        vdir.mkdir(parents=True)
        _copy((run if v.report_key is None else exports) / v.source, vdir)
        shutil.copy2(exports / SIGNS_FILE, vdir / SIGNS_FILE)
        (vdir / "model-instance-metadata.json").write_text(json.dumps({
            "ownerSlug": owner, "modelSlug": MODEL_SLUG, "instanceSlug": v.slug,
            "framework": v.framework, "overview": f"{TITLE}: {v.format_notes.split('.')[0]}.",
            "usage": instance_usage(v), "licenseName": LICENSE,
            "fineTunable": v.framework == "pyTorch", "trainingData": TRAINING_DATA,
            "modelInstanceType": "Unspecified", "baseModelInstanceId": 0, "externalBaseModelUrl": "",
        }, indent=2))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", required=True, help="Kaggle username")
    p.add_argument("--out", type=Path, default=ROOT / "kaggle" / "model")
    p.add_argument("--exports", type=Path, default=ROOT / "exports" / "transformer")
    p.add_argument("--run", type=Path, default=ROOT / "runs" / "transformer_seq_aug")
    p.add_argument("--public", action="store_true", help="publish the model publicly")
    a = p.parse_args()
    print(f"Wrote {build(a.out, a.owner, a.exports, a.run, a.public)}")


if __name__ == "__main__":
    main()
