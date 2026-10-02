"""Generate two R kernels that read the results dataset: an R notebook and an R Markdown report.

    python kaggle/build_r_notebooks.py --owner <kaggle-username> [--public]
    kaggle kernels push -p kaggle/r/notebook
    kaggle kernels push -p kaggle/r/markdown
"""

import argparse
import json
from pathlib import Path

from build_dataset import DATASET_SLUG

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = "/kaggle/input"
KEYWORDS = ["classification"]

FIND_DATA = f"""results <- dirname(list.files("{INPUT_DIR}", pattern = "^epochs\\\\.csv$", recursive = TRUE, full.names = TRUE)[1])"""

NOTEBOOK_SLUG = "asl-realtime-training-curves-in-r"
NOTEBOOK_TITLE = "ASL Realtime - Training Curves in R"
NOTEBOOK_CELLS = [
    ("markdown", f"""# {NOTEBOOK_TITLE}

Eight training runs of small models that recognize 250 isolated ASL signs from MediaPipe
landmarks: a GRU, several Conv1D variants and a temporal transformer with three seeds. The numbers
come from the dataset [`{DATASET_SLUG}`](https://www.kaggle.com/datasets/OWNER/{DATASET_SLUG}).
Accuracy is always measured on signers that were not in the training set."""),
    ("code", f"""library(ggplot2)
library(dplyr)
library(readr)

{FIND_DATA}
runs <- read_csv(file.path(results, "runs.csv"), show_col_types = FALSE)
epochs <- read_csv(file.path(results, "epochs.csv"), show_col_types = FALSE) |>
  left_join(select(runs, run, model), by = "run")
colors <- c(gru = "#8a8fa3", conv1d = "#4c9be8", transformer = "#f2a541")
glimpse(runs)"""),
    ("markdown", "## Validation accuracy per epoch\n\nRuns with augmentation train for 60 epochs, the rest for 30."),
    ("code", """ggplot(epochs, aes(epoch, val_top1 * 100, group = run, color = model)) +
  geom_line(linewidth = 0.9) +
  scale_color_manual(values = colors) +
  coord_cartesian(ylim = c(40, 78)) +
  labs(x = "epoch", y = "top-1 on held-out signers (%)", color = NULL) +
  theme_minimal(base_size = 14)"""),
    ("markdown", "## Is the model overfitting?\n\nTraining loss includes label smoothing and augmentation, "
                 "so it sits above the validation loss for the augmented runs."),
    ("code", """epochs |>
  filter(run %in% c("conv1d", "conv1d_seq_taug", "transformer_seq_aug")) |>
  tidyr::pivot_longer(c(train_loss, val_loss), names_to = "split", values_to = "loss") |>
  ggplot(aes(epoch, loss, color = split)) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~run) +
  labs(x = "epoch", y = "loss", color = NULL) +
  theme_minimal(base_size = 14)"""),
    ("markdown", "## Best accuracy of each run"),
    ("code", """runs |>
  mutate(run = reorder(run, val_top1)) |>
  ggplot(aes(val_top1 * 100, run, fill = model)) +
  geom_col() +
  geom_text(aes(label = sprintf("%.1f", val_top1 * 100)), hjust = -0.15) +
  scale_fill_manual(values = colors) +
  coord_cartesian(xlim = c(60, 78)) +
  labs(x = "top-1 on held-out signers (%)", y = NULL, fill = NULL) +
  theme_minimal(base_size = 14)"""),
    ("markdown", "## Seed-to-seed spread of the transformer"),
    ("code", """runs |>
  filter(model == "transformer") |>
  summarise(seeds = n(), mean_top1 = mean(val_top1), sd_top1 = sd(val_top1),
            mean_top5 = mean(val_top5), sd_top5 = sd(val_top5))"""),
]

MARKDOWN_SLUG = "asl-realtime-export-size-vs-accuracy"
MARKDOWN_TITLE = "ASL Realtime - Export Size vs Accuracy"
MARKDOWN = f"""---
title: "{MARKDOWN_TITLE}"
output: html_document
---

```{{r setup, include=FALSE}}
knitr::opts_chunk$set(echo = TRUE, message = FALSE, warning = FALSE)
library(ggplot2)
library(dplyr)
library(readr)
```

A transformer that recognizes 250 ASL signs was exported to Core ML and TFLite, each with full
and 8-bit weights. This report asks what quantization costs. The numbers come from `exports.csv`
in the dataset [`{DATASET_SLUG}`](https://www.kaggle.com/datasets/OWNER/{DATASET_SLUG}); every
export was checked on all 14,248 validation clips.

```{{r load}}
{FIND_DATA}
exports <- read_csv(file.path(results, "exports.csv"), show_col_types = FALSE) |>
  mutate(runtime = ifelse(grepl("^coreml", variant), "Core ML", "TFLite"),
         weights = sub(".*_", "", variant))
knitr::kable(select(exports, variant, size_mb, top1, torch_top1, agreement, host_ms_median), digits = 4)
```

## Size against accuracy

```{{r size-accuracy, fig.width=8, fig.height=4.5}}
ggplot(exports, aes(size_mb, top1 * 100, color = runtime, label = weights)) +
  geom_hline(aes(yintercept = torch_top1 * 100), linetype = "dashed", color = "grey50") +
  geom_point(size = 4) +
  geom_text(vjust = -1.2, show.legend = FALSE) +
  scale_color_manual(values = c("Core ML" = "#4c9be8", "TFLite" = "#f2a541")) +
  expand_limits(x = 0, y = c(74, 75.2)) +
  labs(x = "size (MB)", y = "top-1 on held-out signers (%)", color = NULL,
       caption = "Dashed line: the PyTorch checkpoint") +
  theme_minimal(base_size = 14)
```

## What 8-bit weights cost

```{{r cost}}
exports |>
  group_by(runtime) |>
  summarise(size_saved = 1 - min(size_mb) / max(size_mb),
            top1_lost_points = 100 * (max(top1) - min(top1)),
            agreement_8bit = min(agreement)) |>
  knitr::kable(digits = 3)
```

`size_saved` is the share of the file size that 8-bit weights remove. `agreement_8bit` is how
often the 8-bit export picks the same sign as the PyTorch checkpoint.
"""


def _notebook(owner: str) -> dict:
    cells = []
    for kind, source in NOTEBOOK_CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.replace("OWNER", owner).splitlines(keepends=True)}
        if kind == "code":
            cell |= {"execution_count": None, "outputs": []}
        cells.append(cell)
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "ir", "display_name": "R", "language": "R"},
                         "language_info": {"name": "R"}}}


def _metadata(owner: str, slug: str, title: str, code_file: str, language: str, kind: str, public: bool) -> dict:
    return {
        "id": f"{owner}/{slug}", "title": title, "code_file": code_file,
        "language": language, "kernel_type": kind, "is_private": str(not public).lower(),
        "enable_gpu": "false", "enable_internet": "false",
        "dataset_sources": [f"{owner}/{DATASET_SLUG}"], "competition_sources": [],
        "kernel_sources": [], "model_sources": [], "keywords": KEYWORDS,
    }


def build(out_dir: Path, owner: str, public: bool = False) -> Path:
    nb_dir, md_dir = out_dir / "notebook", out_dir / "markdown"
    nb_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    (nb_dir / f"{NOTEBOOK_SLUG}.ipynb").write_text(json.dumps(_notebook(owner), indent=1))
    (nb_dir / "kernel-metadata.json").write_text(json.dumps(_metadata(
        owner, NOTEBOOK_SLUG, NOTEBOOK_TITLE, f"{NOTEBOOK_SLUG}.ipynb", "r", "notebook", public), indent=2))
    (md_dir / f"{MARKDOWN_SLUG}.Rmd").write_text(MARKDOWN.replace("OWNER", owner))
    (md_dir / "kernel-metadata.json").write_text(json.dumps(_metadata(
        owner, MARKDOWN_SLUG, MARKDOWN_TITLE, f"{MARKDOWN_SLUG}.Rmd", "rmarkdown", "script", public), indent=2))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--owner", required=True, help="Kaggle username")
    p.add_argument("--out", type=Path, default=ROOT / "kaggle" / "r")
    p.add_argument("--public", action="store_true", help="publish the kernels publicly")
    a = p.parse_args()
    print(f"Wrote {build(a.out, a.owner, a.public)}")


if __name__ == "__main__":
    main()
