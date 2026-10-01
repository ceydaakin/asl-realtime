# ASL Realtime

Real-time American Sign Language recognition from hand and pose landmarks with a temporal transformer. The model runs on-device on a phone.

> **Status:** The temporal transformer reaches 74.7% top-1 on unseen signers and 55.1% on WLASL videos it was never trained on. It exports to Core ML and TFLite at 1.4 MB and runs in a browser demo. Still open: latency on real phones and a native app.

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
- **Augmentation:** scale, rotation, shear, landmark dropout, speed change, frame dropout. No mirroring: clips are already mirrored so the dominant hand is always on the same side
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
| **Transformer** | **sequence** | ✓ + time | 60 | 1.26 M | **74.7%** | 92.3% | 74.6% | M4 Pro |
| Conv1D | sequence | ✓ + time | 60 | 0.51 M | 72.8% | 92.1% | 72.8% | M4 Pro |
| Conv1D | global | – | 30 | 0.51 M | 69.9% | 90.4% | 69.8% | M4 Pro |
| GRU | global | – | 30 | 0.89 M | 68.4% | 90.3% | 68.4% | M4 Pro |
| Conv1D | sequence | – | 30 | 0.51 M | 70.9% | 90.6% | 70.8% | M4 Pro |
| Conv1D | global | ✓ | 60 | 0.51 M | 71.9% | 91.5% | 71.9% | M4 Pro |
| Conv1D | sequence | ✓ | 60 | 0.51 M | 71.7% | 91.2% | 71.7% | Kaggle T4 |
| GRU | sequence | ✓ | 60 | 0.89 M | 70.5% | 90.7% | 70.4% | Kaggle T4 |

- **Augmentation** (scale, rotation, shear, and dropping the lip or arm points) adds about 2 points.
- **Sequence normalization** (center and scale per clip) adds 1 point without augmentation and is on par with global normalization when augmentation is on. It stays the default anyway. The val signers are all framed similarly, so this table can't show how much it matters. A live camera varies position and distance, and sequence normalization is what handles that.
- Differences under about 0.5 points are within run-to-run noise: the same code run on MPS vs CUDA moves results by that much. Multi-seed runs are still to do.
- Conv1D beats GRU in every setting, with 57% of the parameters.
- **Temporal augmentation** ("+ time": random signing speed and dropped frames) adds about 1 point to Conv1D.
- The **transformer** (4 layers, 192 wide) adds 2 more points for 2.5× the parameters. It is the exported model.
- All numbers are single-seed runs.

### Cross-dataset (WLASL)

The same checkpoints on WLASL videos of the 200 signs both datasets share, with no fine-tuning. The videos go through MediaPipe Holistic and [`live.py`](src/asl_realtime/live.py), the same path a live camera takes. 831 clips: of 1,734 direct-download links, the rest are dead or show no hand. YouTube links are skipped.

| Model | Norm | Augment | Top-1 | Top-5 |
|---|---|---|---|---|
| **Transformer** | sequence | ✓ + time | **55.1%** | 75.3% |
| Conv1D | sequence | ✓ + time | 51.4% | 75.1% |
| Conv1D | sequence | ✓ | 51.9% | 75.2% |
| Conv1D | global | ✓ | 47.9% | 70.6% |

Sequence normalization is worth 4 points here, which the GISLR validation split could not show. The 20-point drop from GISLR comes from different signers, studio framing, sign variants that differ between the datasets, and two-handed signs.

```bash
python -m asl_realtime.wlasl build --index WLASL_v0.3.json --out data/wlasl
python -m asl_realtime.wlasl eval --clips data/wlasl/clips.npz --checkpoint runs/transformer/model.pt
```

### Export

The transformer, checked on all 14,248 validation clips. "Same as PyTorch" is how often the exported model picks the same sign.

| Format | Weights | Size | Top-1 | Same as PyTorch |
|---|---|---|---|---|
| Core ML | float16 | 2.61 MB | 74.8% | 99.7% |
| Core ML | 8-bit | 1.36 MB | 74.6% | 99.1% |
| TFLite | float32 | 5.19 MB | 74.7% | 100% |
| TFLite | 8-bit | 1.50 MB | 74.6% | 98.8% |

One prediction takes about 0.5 ms on an M4 Pro CPU and 3.5 ms in desktop Chromium (WebAssembly, 8-bit TFLite). **Phone latency is not measured yet.**

```bash
pip install -e ".[export]"   # needs torch<2.14, see pyproject.toml
python -m asl_realtime.export --checkpoint runs/transformer/model.pt --out exports/transformer
```

[Kaggle notebook](https://www.kaggle.com/code/ceydaakin2004/asl-realtime-landmark-baselines) (currently private).

```bash
python -m asl_realtime.train --model transformer --norm sequence --augment --epochs 60
```

## Demo

**Browser** ([`app/web`](app/web)): MediaPipe Holistic and the TFLite model run in the page, on a phone or a laptop. No video is uploaded. The Benchmark button times the model on that device.

```bash
mkdir -p app/web/model
cp exports/transformer/asl_int8.tflite app/web/model/asl.tflite
cp exports/transformer/signs.json app/web/model/
python -m http.server -d app/web 8000   # then open http://localhost:8000
```

A phone only allows camera access over HTTPS, so serve the folder from an HTTPS host to try it there. It has been tested with video files in desktop Chromium, not yet with a live camera or on a phone.

**Python** (webcam or video file):

```bash
pip install -e ".[demo]"
curl -LO https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
python -m asl_realtime.demo --checkpoint runs/transformer/model.pt
```

Raise your hand, sign, lower it: the sign is classified when the hand leaves the picture.

## Roadmap

- [x] Data loading and signer-disjoint splits
- [x] Baseline models (GRU, 1D-CNN)
- [x] Position/scale-invariant normalization (per-clip centering and scaling, inside the model)
- [x] Augmentation (affine, landmark-group dropout)
- [x] Temporal augmentation (speed change, frame dropout)
- [x] Temporal transformer
- [x] Quantization and Core ML / TFLite export
- [x] Cross-dataset evaluation on WLASL
- [x] Live preprocessing and real-time demo (Python and browser)
- [ ] Try the browser demo on a phone with a live camera
- [ ] Native mobile app (Core ML on iOS, TFLite on Android)
- [ ] On-device benchmark on iPhone / Android
- [ ] Multi-seed runs

## Project structure

```
src/asl_realtime/
  data.py, features.py, landmarks.py, labels.py   # dataset, landmark layout, sign names
  normalize.py, augment.py, models.py, train.py   # training
  export.py                                       # Core ML / TFLite conversion and parity check
  live.py, demo.py                                # camera frames -> model input, webcam demo
  wlasl.py                                        # cross-dataset evaluation
app/web/                                          # browser demo
kaggle/                                           # notebook builder
```

## License

MIT
