"""Stage the experiment results as a Kaggle dataset (CSV files with documented columns).

    python kaggle/build_dataset.py --owner <kaggle-username> [--public]
    kaggle datasets create -p kaggle/dataset
    kaggle datasets metadata --update -p kaggle/dataset <owner>/asl-realtime-experiment-results
    kaggle datasets version -p kaggle/dataset -m "<what changed>"

`metadata --update` sets the sources and the cover image, which `create` ignores. File and column
descriptions only took effect after a `version`.
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_SLUG = "asl-realtime-experiment-results"
TITLE = "ASL Realtime Experiment Results"
SUBTITLE = "Training curves and export checks for landmark-based ASL sign recognition"
# `datasets metadata --update` wants the display name; the one-time `datasets create` wants "CC-BY-4.0".
LICENSE = "Attribution 4.0 International (CC BY 4.0)"
KEYWORDS = ["deep learning", "classification", "computer vision", "model comparison"]
REPO = "https://github.com/ceydaakin/asl-realtime"
COVER = ROOT / "kaggle" / "assets" / "dataset-cover.png"
SOURCES = (f"Produced by the training and export code in {REPO}, from the Google Isolated Sign Language "
           "Recognition landmarks (Kaggle dataset markwijkhuizen/gislr-dataset-public).")

# Finished runs to publish, and whether they used temporal augmentation (not stored in metrics.json).
RUNS = {
    "gru": False, "conv1d": False, "conv1d_seq": False, "conv1d_global_aug": False,
    "conv1d_seq_taug": True, "transformer_seq_aug": True,
    "transformer_seq_aug_s1": True, "transformer_seq_aug_s2": True,
}

# file -> (description, [(column, type, description)])
SCHEMA = {
    "runs.csv": ("One row per training run: configuration and the best and last validation scores.", [
        ("run", "string", "Run name; joins to epochs.csv"),
        ("model", "string", "Architecture: gru, conv1d or transformer"),
        ("norm", "string", "Landmark normalization: global (dataset statistics) or sequence (per clip)"),
        ("augment", "boolean", "Affine and landmark-dropout augmentation was on"),
        ("temporal_augment", "boolean", "Random signing speed and frame dropout were on"),
        ("params", "integer", "Number of trainable parameters"),
        ("epochs", "integer", "Epochs trained"),
        ("best_epoch", "integer", "Epoch with the highest validation top-1 accuracy"),
        ("val_top1", "number", "Top-1 accuracy on held-out signers at the best epoch (0 to 1)"),
        ("val_top5", "number", "Top-5 accuracy on held-out signers at the best epoch (0 to 1)"),
        ("last_val_top1", "number", "Top-1 accuracy on held-out signers after the last epoch (0 to 1)"),
        ("last_val_top5", "number", "Top-5 accuracy on held-out signers after the last epoch (0 to 1)"),
        ("n_train", "integer", "Training clips"),
        ("n_val", "integer", "Validation clips, all from participants not in the training set"),
        ("device", "string", "Device the run trained on"),
    ]),
    "epochs.csv": ("One row per run and epoch: the training curve.", [
        ("run", "string", "Run name; joins to runs.csv"),
        ("epoch", "integer", "Epoch number, starting at 1"),
        ("train_loss", "number", "Mean training loss (cross-entropy with label smoothing 0.1)"),
        ("val_loss", "number", "Mean validation loss"),
        ("val_top1", "number", "Top-1 accuracy on held-out signers (0 to 1)"),
        ("val_top5", "number", "Top-5 accuracy on held-out signers (0 to 1)"),
        ("seconds", "number", "Wall-clock seconds for the epoch"),
    ]),
    "exports.csv": ("One row per exported format of the transformer, checked on all validation clips.", [
        ("variant", "string", "Export format and weight precision"),
        ("file", "string", "File name of the exported model"),
        ("size_mb", "number", "Size on disk in megabytes"),
        ("top1", "number", "Top-1 accuracy of the exported model (0 to 1)"),
        ("torch_top1", "number", "Top-1 accuracy of the PyTorch checkpoint it was exported from (0 to 1)"),
        ("agreement", "number", "Share of clips where the export and the checkpoint pick the same sign (0 to 1)"),
        ("max_prob_diff", "number", "Largest absolute difference between any two output probabilities"),
        ("host_ms_median", "number", "Median milliseconds per prediction on an Apple M4 Pro CPU"),
    ]),
    "signs.csv": ("The 250 sign classes in model output order.", [
        ("index", "integer", "Class index in the model output"),
        ("sign", "string", "English gloss of the ASL sign"),
    ]),
}

DESCRIPTION = f"""Results of training small landmark-based models to recognize 250 isolated American Sign
Language signs, from the [asl-realtime]({REPO}) project. Use it to compare architectures,
normalization and augmentation choices, or to plot training curves without rerunning anything.

## Files

- `runs.csv`: one row per training run (GRU, Conv1D and temporal transformer; three transformer seeds)
- `epochs.csv`: per-epoch loss and accuracy for each run
- `exports.csv`: Core ML and TFLite exports of the transformer, with size, accuracy and agreement with PyTorch
- `signs.csv`: the 250 sign names in class-index order

## How the numbers were produced

Models were trained on MediaPipe landmarks from the Google Isolated Sign Language Recognition
data (`markwijkhuizen/gislr-dataset-public`), 80,229 training clips. Every accuracy is measured
on 14,248 clips from **participants who are not in the training set**, so it describes unseen
signers. Chance is 0.4%. Differences under about 0.5 points are within run-to-run noise.

The trained transformer itself is published as the Kaggle model `asl-realtime-transformer`.
The training code is in the linked repository.
"""


def _run_row(name: str, m: dict) -> dict:
    return {"run": name, "norm": m.get("norm", "global"), "augment": m.get("augment", False),
            "temporal_augment": RUNS[name], **{k: m[k] for k in (
                "model", "params", "epochs", "best_epoch", "val_top1", "val_top5",
                "last_val_top1", "last_val_top5", "n_train", "n_val", "device")}}


def _write(path: Path, rows: list[dict]) -> None:
    columns = [c for c, _, _ in SCHEMA[path.name][1]]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def metadata(owner: str, public: bool, image: str) -> dict:
    return {
        "title": TITLE, "subtitle": SUBTITLE, "id": f"{owner}/{DATASET_SLUG}",
        "description": DESCRIPTION, "isPrivate": not public,
        "licenses": [{"name": LICENSE}], "keywords": KEYWORDS,
        "userSpecifiedSources": SOURCES, "expectedUpdateFrequency": "never", "image": image,
        "resources": [
            {"path": name, "description": description,
             "schema": {"fields": [{"name": c, "type": t, "description": d} for c, t, d in fields]}}
            for name, (description, fields) in SCHEMA.items()
        ],
    }


def build(out_dir: Path, owner: str, runs: Path, exports: Path, public: bool = False) -> Path:
    metrics = {name: json.loads((runs / name / "metrics.json").read_text()) for name in RUNS}
    report = json.loads((exports / "report.json").read_text())
    signs = json.loads((exports / "signs.json").read_text())
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    _write(out_dir / "runs.csv", [_run_row(name, m) for name, m in metrics.items()])
    _write(out_dir / "epochs.csv", [{"run": name, **e} for name, m in metrics.items() for e in m["history"]])
    _write(out_dir / "exports.csv", [{"variant": k, **v} for k, v in report["variants"].items()])
    _write(out_dir / "signs.csv", [{"index": i, "sign": s} for i, s in enumerate(signs)])
    (out_dir / "dataset-metadata.json").write_text(json.dumps(metadata(owner, public, os.path.relpath(COVER, out_dir)), indent=2))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", required=True, help="Kaggle username")
    p.add_argument("--out", type=Path, default=ROOT / "kaggle" / "dataset")
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    p.add_argument("--exports", type=Path, default=ROOT / "exports" / "transformer")
    p.add_argument("--public", action="store_true", help="publish the dataset publicly")
    a = p.parse_args()
    print(f"Wrote {build(a.out, a.owner, a.runs, a.exports, a.public)}")


if __name__ == "__main__":
    main()
