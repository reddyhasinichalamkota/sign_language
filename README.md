# Indian Sign Language Detection

Real-time **Indian Sign Language (ISL)** recognition from a standard webcam, built with MediaPipe hand tracking and deep learning in an OpenCV desktop app.

- **Static signs:** A–Z alphabet and finger-counted numbers 0–5
- **Dynamic signs:** 61 words and phrases (*Hello*, *Thank you*, *Good Morning*, ...)
- **Runs offline on CPU:** no GPU, server or internet needed

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-1.1-0097A7)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12-5C3EE8?logo=opencv&logoColor=white)
![Tests](https://img.shields.io/badge/tests-76%20passing-2EA44F)

---

## Features

- Live A–Z recognition, one- and two-handed signs
- Number mode: counts raised fingers (0–5)
- Dynamic recognition: classifies the last ~4 seconds of video into 61 signs
- Sign shortcuts: hold **W** (WhatsApp), **Y** (YouTube) or **L** (Google) for 1 second
- Guided A–Z and 0–5 webcam self-tests that save results to CSV
- Dataset integrity checks for duplicates, label conflicts and leakage

## How It Works

### Static signs (A–Z)

```
Webcam frame ─► MediaPipe hand landmarks ─┬─► Geometric rule classifier ─┐
                                          └─► Hand crop ─► CNN ──────────┴─► CNN-first fusion + smoothing ─► Letter
```

- **Hand tracking:** MediaPipe Gesture Recognizer (`gesture_recognizer.task`), up to two hands.
- **CNN:** exported to TensorFlow Lite (`model/custom_cnn.tflite`). Input 224×224 RGB, output 26-way softmax (A–Z).
- **Rules:** hand geometry gives a fallback prediction for some letters (e.g. open hand → B, fist → A).
- **Fusion (CNN first):** if the CNN is at least 50% confident its letter is used; the rules only apply when the CNN is unsure.
- **Smoothing:** predictions are averaged over a short window for a stable output.

### Dynamic signs (words and phrases)

```
Last ~4 s of video ─► 16 sampled frames ─► MobileNetV2 (frozen) ─► 2-layer Bi-GRU ─► Softmax (61 signs)
```

- **Default model:** `models/dynamic/dynamic_model.keras` (MobileNetV2 features → Bi-GRU 128 → 64 units).
- **Alternative:** one-shot matching with dynamic time warping against one reference video per sign (`models/dynamic_oneshot/`).

## Results

| Model | Evaluation | Result |
| :--- | :--- | :--- |
| Dynamic: MobileNetV2 + Bi-GRU | Held-out test set, 140 clips, 61 classes | **99.3% accuracy** (139/140), macro-F1 **0.978** |
| Dynamic: one-shot (DTW) | 240 clips, 61 classes | 24.2% top-1 (chance 1.7%) |

Dynamic data: 1,043 clips split by recording into 760 train / 143 validation / 140 test. Full metrics are in
[`evaluation_results/dynamic_trained/`](evaluation_results/dynamic_trained/).

> The test clips share signers and environment with the training data, so 99.3% is within-session performance.
> Expect lower accuracy for new signers, lighting and backgrounds.

## Getting Started

**Requirements:** Windows 10/11, macOS or Linux (verified on Windows 11), Python 3.13, a webcam.

```bash
git clone https://github.com/saisharatkaruturi/Indian-Sign-Language-Detection.git
cd Indian-Sign-Language-Detection

python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source .venv/bin/activate         # macOS / Linux

python -m pip install -r requirements.txt
```

The first launch can take 1–3 minutes while TensorFlow and MediaPipe initialise.

## Usage

Run from the repository root.

```bash
python src/main.py                       # interactive menu
python src/main.py --mode static         # live A–Z recognition
python src/main.py --mode dynamic        # live word/phrase recognition
python src/main.py --mode camera_test    # check the webcam
python src/main.py --mode letter_test    # guided A–Z test
python src/main.py --mode number_test    # guided 0–5 test
```

| Key | Action |
| :---: | :--- |
| `Q` | Quit |
| `M` | Toggle mirrored view |
| `C` | Clear recognised sequence |
| `N` | Toggle number mode (0–5) |

Settings (camera, dynamic method, confidence threshold, shortcuts) are in [`config.yaml`](config.yaml).

## Testing

```bash
python -m pytest -v            # unit, model and integration tests
python integration_tests.py    # preprocessing-contract tests only
```

## Data Quality Checks

Needs the static dataset at `Dataset/Training`, `Dataset/Validation` and `Dataset/Testing` (one folder per letter).

```bash
python check_leak.py     # images per split and identical images shared across splits
python check_dupes.py    # duplicates, same image under different letters, leakage -> dupes_report.csv
```

## Retraining the Models

### Static A–Z model

```bash
python train_fast.py                                        # fast: frozen MobileNetV2 features + small classifier
python train_static_v2.py --head-epochs 5 --finetune-epochs 8   # full: head training + fine-tuning + augmentation
python diagnose_cnn.py --model model\custom_cnn_new.tflite
```

| Script | Approach |
| :--- | :--- |
| `train_fast.py` | Extracts MobileNetV2 features once, then trains a small classifier. **Stage A** trains on Training only and reports held-out Validation/Testing accuracy. **Stage B** retrains on all splits for export |
| `train_static_v2.py` | MobileNetV2 head training, then fine-tuning of the top 40 layers, with zoom/shift/rotation/brightness/contrast/RGB↔BGR augmentation |

Both write `model/custom_cnn_new.tflite` and never overwrite `model/custom_cnn.tflite`. To use it, back up the old
model and copy the new one over it. The export takes raw 0–255 float32 pixels.

> The model exported by `train_fast.py` has seen the Testing images (Stage B), so only its Stage A numbers are held-out.

### Dynamic model

Place `Video_Dataset/` in the repository root, then:

```bash
python src/training/train_dynamic.py
python src/training/build_oneshot_gallery.py
python src/training/evaluate_oneshot.py
```

## Project Structure

```
├── src/                     # main.py, camera, dataset, inference, models, preprocessing, training, utils
├── model/                   # custom_cnn.tflite (static A–Z CNN)
├── models/                  # static mapping, dynamic Bi-GRU model, one-shot gallery
├── Dataset/                 # static image dataset (not included)
├── Sample Videos/           # one reference clip per dynamic sign (61)
├── evaluation_results/      # saved metrics and live-test results
├── data_pipeline/           # reusable preprocessing contracts
├── integration/             # mock recognisers for contract tests
├── tests/                   # pytest suite
├── train_fast.py            # fast static retraining
├── train_static_v2.py       # static retraining with fine-tuning
├── diagnose_cnn.py          # static model diagnostics
├── check_leak.py            # cross-split leakage check
├── check_dupes.py           # duplicate and label-conflict report
├── patch_translator.py      # applies the CNN-first fusion fix
├── gesture_recognizer.task  # MediaPipe hand-tracking model
├── config.yaml              # runtime configuration
├── integration_tests.py
├── pytest.ini
└── requirements.txt
```

## Limitations

- Signer independence is not measured; dynamic test clips share signers with the training data.
- The dynamic model used 1,043 of 2,723 available clips, with 1–8 test clips per sign.
- One-shot matching is weak (24.2% top-1) and is only a fallback.
- Numbers 6–9 (two-handed in ISL) are not supported.
- Rule-based confidences are fixed values, not calibrated probabilities.
- Static accuracy is not measured in this repository by default because the image dataset is not included;
  reproduce it with `train_fast.py` (Stage A) or `train_static_v2.py`.
- Static datasets can contain duplicates or train/test leakage. Run the data quality checks before trusting any accuracy figure.

## Troubleshooting

| Problem | Solution |
| :--- | :--- |
| `ModuleNotFoundError: mediapipe` | Activate the virtual environment and reinstall `requirements.txt` |
| `cv2` import errors | Install only `opencv-contrib-python`; uninstall `opencv-python` |
| Camera does not open | Close other apps using the webcam, or change `camera.index` in `config.yaml` |
| `Static CNN not loaded` | Check that `model/custom_cnn.tflite` and `models/static/class_mapping.json` exist |
| Letters like A, B, D, V predicted wrongly | Run `python patch_translator.py` (backup saved as `gesture_translator.py.bak`) |
| `Folder not found` when training | Put the dataset in `Dataset/Training`, `Validation`, `Testing` and run from the repo root |
