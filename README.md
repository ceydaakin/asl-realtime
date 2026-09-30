# ASL Realtime

Real-time American Sign Language recognition from hand and pose landmarks with a temporal transformer. The model runs on-device on a phone.

> **Status:** Early development. The data pipeline, baselines and augmentation are done (71.7% top-1 on unseen signers). A stronger model and the mobile export come next.

## Goal

Recognize isolated ASL signs from a live camera feed with low latency, entirely on-device. No video leaves the phone.

## Approach

```
Camera ─► Landmark extraction (hands + pose + face) ─► Normalization
       ─► Temporal transformer ─► Sign class + confidence
```

- **Features:** 3D landmarks instead of raw pixels, so the model is small, fast and less sensitive to background and skin tone
- **Normalization:** center and scale relative to the shoulders; handle frames with missing hands
- **Model:** a lightweight transformer / 1D-conv encoder over landmark sequences
- **Augmentation:** mirroring, rotation, time warping, landmark dropout
- **Deployment:** export to Core ML (iOS) and TFLite (Android); measure latency on real devices

## Data

- [Google – Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) (250 signs, landmark format)
- [WLASL](https://dxli94.github.io/WLASL/) for cross-dataset generalization

Development currently uses a preprocessed version of the competition data,
[`markwijkhuizen/gislr-dataset-public`](https://www.kaggle.com/datasets/markwijkhuizen/gislr-dataset-public):

| Split | Samples | Shape |
|---|---|---|
| train | 80,229 | 64 frames × 66 landmarks × (x, y, z) |
| val | 14,248 | same, from **held-out participants** |

The 66 landmarks per frame are 40 lip points, the 21 points of the dominant hand and 5 points of the dominant arm. Right-dominant signers are mirrored so every sample looks left-dominant. See [`landmarks.py`](src/asl_realtime/landmarks.py).

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
kaggle auth login
for f in X_train.npy X_val.npy y_train.npy y_val.npy NON_EMPTY_FRAME_IDXS_TRAIN.npy NON_EMPTY_FRAME_IDXS_VAL.npy; do
  kaggle datasets download markwijkhuizen/gislr-dataset-public -f "$f" -p data/gislr_public --unzip
done
pytest
```

## Evaluation

| Metric | Description |
|---|---|
| Top-1 / Top-5 accuracy | Held-out signers only (signer-disjoint split) |
| Cross-dataset accuracy | Train on one dataset, test on the other |
| Latency | ms per prediction on iPhone / Android |
| Model size | MB after quantization |

## Results

Validation on **held-out participants**. There are 250 classes, so chance is 0.4%. x/y landmarks only.

| Model | Norm | Augment | Epochs | Params | Top-1 | Top-5 | Top-1 (last epoch) | Where |
|---|---|---|---|---|---|---|---|---|
| Conv1D | global | – | 30 | 0.51 M | 69.9% | 90.4% | 69.8% | M4 Pro |
| GRU | global | – | 30 | 0.89 M | 68.4% | 90.3% | 68.4% | M4 Pro |
| Conv1D | sequence | – | 30 | 0.51 M | 70.9% | 90.6% | 70.8% | M4 Pro |
| Conv1D | global | ✓ | 60 | 0.51 M | 71.9% | 91.5% | 71.9% | M4 Pro |
| **Conv1D** | **sequence** | ✓ | 60 | 0.51 M | **71.7%** | 91.2% | 71.7% | Kaggle T4 |
| GRU | sequence | ✓ | 60 | 0.89 M | 70.5% | 90.7% | 70.4% | Kaggle T4 |

- **Augmentation** (scale, rotation, shear, and dropping the lip or arm points) adds about 2 points.
- **Sequence normalization** (center and scale per clip) adds 1 point without augmentation and is on par with global normalization when augmentation is on. It stays the default anyway. The val signers are all framed similarly, so this table can't show how much it matters. A live camera varies position and distance, and sequence normalization is what handles that.
- Differences under about 0.5 points are within run-to-run noise: the same code run on MPS vs CUDA moves results by that much. Multi-seed runs are still to do.
- Conv1D beats GRU in every setting, with 57% of the parameters. It is the export candidate.

[Kaggle notebook](https://www.kaggle.com/code/ceydaakin2004/asl-realtime-landmark-baselines) (currently private).

```bash
python -m asl_realtime.train --model conv1d --norm sequence --augment --epochs 60
```

## Roadmap

- [x] Data loading and signer-disjoint splits
- [x] Baseline models (GRU, 1D-CNN)
- [x] Position/scale-invariant normalization (per-clip centering and scaling, inside the model)
- [x] Augmentation (affine, landmark-group dropout)
- [ ] Temporal augmentation (time warping, frame dropout)
- [ ] Temporal transformer
- [ ] Quantization and Core ML / TFLite export
- [ ] Real-time mobile demo app
- [ ] On-device benchmark

## Project structure (planned)

```
data/         # loaders, splits, augmentation
models/       # baselines + transformer
train/        # training and evaluation scripts
export/       # Core ML / TFLite conversion
app/          # mobile demo
```

## License

MIT
