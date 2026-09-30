# ASL Realtime

Real-time American Sign Language recognition from hand and pose landmarks with a temporal transformer. The model runs on-device on a phone.

> **Status:** Planning. No code yet.

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

## Evaluation

| Metric | Description |
|---|---|
| Top-1 / Top-5 accuracy | Held-out signers only (signer-disjoint split) |
| Cross-dataset accuracy | Train on one dataset, test on the other |
| Latency | ms per prediction on iPhone / Android |
| Model size | MB after quantization |

## Roadmap

- [ ] Data loading and signer-disjoint splits
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
