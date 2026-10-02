"""Generate the Kaggle notebook from the package source.

The notebook inlines src/asl_realtime so Kaggle runs exactly the code in this
repo, without needing internet access or a pip install.

    python kaggle/build_notebook.py --owner <kaggle-username>
    kaggle kernels push -p kaggle/kernel
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "asl_realtime"
MODULE_ORDER = ["landmarks", "data", "features", "normalize", "augment", "models", "metrics", "train", "errors"]
DATASET = "markwijkhuizen/gislr-dataset-public"
KERNEL_SLUG = "asl-realtime-landmark-baselines"
TITLE = "ASL Realtime - Landmark Baselines"

INTRO = f"""# {TITLE}

Baseline models for isolated American Sign Language recognition from MediaPipe landmarks
(250 signs, Google ISLR data). The validation split uses **held-out participants**, so the
accuracy below measures how well the model handles signers it has never seen.

This notebook is generated from the project repo (`asl-realtime`), with the package source
inlined below. Data: [`{DATASET}`](https://www.kaggle.com/datasets/{DATASET})."""

FIND_DATA = """from pathlib import Path

DATA_ROOT = next(Path("/kaggle/input").rglob("X_train.npy")).parent
print(DATA_ROOT)"""

RUN = """# name -> TrainConfig overrides
EXPERIMENTS = {
    "conv1d_seq_aug": dict(model="conv1d", norm="sequence", augment=True, epochs=60),
    "gru_seq_aug": dict(model="gru", norm="sequence", augment=True, epochs=60),
}

results = {}
for name, overrides in EXPERIMENTS.items():
    print(f"=== {name}")
    results[name] = run(TrainConfig(data_root=str(DATA_ROOT), out_dir=f"/kaggle/working/runs/{name}", **overrides))"""

SUMMARY = """import pandas as pd

pd.DataFrame([
    {"experiment": name, "params": r["params"], "norm": r["norm"], "augment": r["augment"],
     "val_top1": r["val_top1"], "val_top5": r["val_top5"], "last_val_top1": r["last_val_top1"],
     "best_epoch": r["best_epoch"]}
    for name, r in results.items()
]).set_index("experiment").round(4)"""

_RELATIVE_IMPORT = re.compile(r"^from \.\w* import .*$\n", re.MULTILINE)


def module_source(name: str) -> str:
    src = (SRC / f"{name}.py").read_text()
    src = _RELATIVE_IMPORT.sub("", src)
    cut = min((i for i in (src.find("\ndef main("), src.find('\nif __name__ == "__main__"')) if i != -1),
              default=len(src))
    return f"# --- asl_realtime/{name}.py ---\n{src[:cut].rstrip()}\n"


def _cell(kind: str, source: str, tags: list[str] | None = None) -> dict:
    cell = {"cell_type": kind, "metadata": {"tags": tags or []}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell |= {"execution_count": None, "outputs": []}
    return cell


def notebook() -> dict:
    cells = [_cell("markdown", INTRO), _cell("code", FIND_DATA)]
    cells += [_cell("code", module_source(m), tags=["module"]) for m in MODULE_ORDER]
    cells += [_cell("markdown", "## Train"), _cell("code", RUN),
              _cell("markdown", "## Results"), _cell("code", SUMMARY)]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


def build(out_dir: Path, owner: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = f"{KERNEL_SLUG}.ipynb"
    (out_dir / code_file).write_text(json.dumps(notebook(), indent=1))
    meta = {
        "id": f"{owner}/{KERNEL_SLUG}", "title": TITLE, "code_file": code_file,
        "language": "python", "kernel_type": "notebook", "is_private": "true",
        "enable_gpu": "true", "enable_internet": "false",
        "dataset_sources": [DATASET], "competition_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", required=True, help="Kaggle username")
    p.add_argument("--out", type=Path, default=ROOT / "kaggle" / "kernel")
    a = p.parse_args()
    print(f"Wrote {build(a.out, a.owner)}")


if __name__ == "__main__":
    main()
