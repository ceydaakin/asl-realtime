"""Generate the Kaggle evaluation notebook: the published model on held-out signers, sign by sign.

It uses the three other Kaggle artifacts of this project: the utility script (code), the model
(weights) and the results dataset (training curves).

    python kaggle/build_eval_notebook.py --owner <kaggle-username> [--public]
    kaggle kernels push -p kaggle/eval
"""

import argparse
import json
from pathlib import Path

from build_dataset import DATASET_SLUG
from build_model import MODEL_SLUG
from build_notebook import DATASET, _cell
from build_script import MODULE as LIB
from build_script import SCRIPT_SLUG

ROOT = Path(__file__).resolve().parents[1]
KERNEL_SLUG = "asl-realtime-which-signs-are-hard"
TITLE = "ASL Realtime - Which Signs Are Hard"
KEYWORDS = ["classification", "transformers"]

INTRO = f"""# {TITLE}

A 1.3 M-parameter transformer recognizes 250 isolated ASL signs from MediaPipe landmarks with
about 75% top-1 accuracy on signers it has never seen. This notebook looks at the other 25%:
which signs it misses, and what it mistakes them for.

Everything is loaded from Kaggle, nothing is trained here:

- code: the utility script [`{SCRIPT_SLUG}`](https://www.kaggle.com/code/OWNER/{SCRIPT_SLUG})
- weights: the model [`{MODEL_SLUG}`](https://www.kaggle.com/models/OWNER/{MODEL_SLUG}) (PyTorch variation)
- training curves: the dataset [`{DATASET_SLUG}`](https://www.kaggle.com/datasets/OWNER/{DATASET_SLUG})
- landmarks: [`{DATASET}`](https://www.kaggle.com/datasets/{DATASET})"""

SETUP = f"""import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(next(Path("/kaggle").rglob("{LIB}.py")).parent))
from {LIB} import USE_DIMS, frame_mask, load_model, load_split, per_sign_accuracy, top_confusions

INPUT = Path("/kaggle/input")
DATA_ROOT = next(INPUT.rglob("X_val.npy")).parent
MODEL_PATH = next(INPUT.rglob("model.pt"))
SIGNS = json.loads(next(INPUT.rglob("signs.json")).read_text())
RESULTS = next(INPUT.rglob("epochs.csv")).parent
OUT = Path("/kaggle/working")
print(DATA_ROOT, MODEL_PATH, RESULTS, sep="\\n")"""

PREDICT = """model = load_model(MODEL_PATH)
val = load_split(DATA_ROOT, "val")
xy = torch.from_numpy(np.ascontiguousarray(val.X[..., :USE_DIMS], dtype=np.float32))
mask = torch.from_numpy(frame_mask(val.frame_idxs))
y = np.asarray(val.y, np.int64)

with torch.no_grad():
    logits = torch.cat([model(xy[i:i + 512], mask[i:i + 512]) for i in range(0, len(y), 512)])
top5 = logits.topk(5).indices.numpy()
pred = top5[:, 0]

print(f"{len(y):,} clips from held-out signers")
print(f"top-1 {np.mean(pred == y):.1%}   top-5 {np.mean((top5 == y[:, None]).any(1)):.1%}")"""

PER_SIGN = """per_sign = pd.DataFrame(per_sign_accuracy(y, pred, SIGNS))
per_sign.to_csv(OUT / "per_sign_accuracy.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(per_sign["top1"] * 100, bins=20, color="#4c9be8")
axes[0].set(xlabel="top-1 accuracy of a sign (%)", ylabel="number of signs", title="Accuracy per sign")
worst = per_sign.head(15)
axes[1].barh(worst["sign"], worst["top1"] * 100, color="#f2a541")
axes[1].invert_yaxis()
axes[1].set(xlabel="top-1 accuracy (%)", title="The 15 hardest signs")
plt.tight_layout()
plt.show()
per_sign.head(15)"""

CONFUSIONS = """confusions = pd.DataFrame(top_confusions(y, pred, SIGNS, n=25))
confusions.to_csv(OUT / "confusions.csv", index=False)
confusions"""

CURVES = """epochs = pd.read_csv(RESULTS / "epochs.csv")
runs = pd.read_csv(RESULTS / "runs.csv").set_index("run")
colors = {"gru": "#8a8fa3", "conv1d": "#4c9be8", "transformer": "#f2a541"}

fig, ax = plt.subplots(figsize=(10, 5))
for run, curve in epochs.groupby("run"):
    ax.plot(curve["epoch"], curve["val_top1"] * 100, color=colors[runs.loc[run, "model"]])
ax.legend(handles=[plt.Line2D([], [], color=c, label=m) for m, c in colors.items()])
ax.set(xlabel="epoch", ylabel="top-1 on held-out signers (%)", ylim=(40, 78), title="Training curves of all runs")
plt.show()
runs[["model", "norm", "augment", "temporal_augment", "params", "val_top1", "val_top5"]].sort_values("val_top1")"""


def notebook(owner: str) -> dict:
    cells = [
        _cell("markdown", INTRO.replace("OWNER", owner)), _cell("code", SETUP),
        _cell("markdown", "## Accuracy on unseen signers"), _cell("code", PREDICT),
        _cell("markdown", "## Per sign\n\nSigns are sorted from least to most accurate. "
                          "The table is saved as `per_sign_accuracy.csv`."),
        _cell("code", PER_SIGN),
        _cell("markdown", "## What gets mistaken for what\n\nThe most frequent wrong predictions. `share_of_true` "
                          "is the share of that sign's clips that went to this wrong answer. Saved as `confusions.csv`."),
        _cell("code", CONFUSIONS),
        _cell("markdown", "## How the model got here\n\nValidation accuracy per epoch for every run in the results "
                          "dataset: GRU and Conv1D baselines, then the transformer with three seeds."),
        _cell("code", CURVES),
    ]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


def build(out_dir: Path, owner: str, public: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{KERNEL_SLUG}.ipynb"
    (out_dir / code_file).write_text(json.dumps(notebook(owner), indent=1))
    meta = {
        "id": f"{owner}/{KERNEL_SLUG}", "title": TITLE, "code_file": code_file,
        "language": "python", "kernel_type": "notebook", "is_private": str(not public).lower(),
        "enable_gpu": "false", "enable_internet": "false",
        "dataset_sources": [DATASET, f"{owner}/{DATASET_SLUG}"], "competition_sources": [],
        "kernel_sources": [f"{owner}/{SCRIPT_SLUG}"],
        "model_sources": [f"{owner}/{MODEL_SLUG}/pyTorch/default/1"], "keywords": KEYWORDS,
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", required=True, help="Kaggle username")
    p.add_argument("--out", type=Path, default=ROOT / "kaggle" / "eval")
    p.add_argument("--public", action="store_true", help="publish the notebook publicly")
    a = p.parse_args()
    print(f"Wrote {build(a.out, a.owner, a.public)}")


if __name__ == "__main__":
    main()
