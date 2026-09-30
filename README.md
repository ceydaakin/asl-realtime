# ASL Realtime

Real-time American Sign Language recognition from hand and pose landmarks with a temporal transformer. The model runs on-device on a phone.

> **Status:** Early development. The data pipeline is in place; models come next.

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

## Roadmap

- [x] Data loading and signer-disjoint splits
- [ ] Landmark normalization + augmentation
- [ ] Baseline models (GRU, 1D-CNN)
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
