"""
Find out WHY the static CNN mispredicts (G, H, J, O, T, W, Z ...).

It runs model/custom_cnn.tflite directly on the Testing images, with every combination of
  - colour order : BGR (what cv2.imread gives) vs RGB
  - scaling      : raw 0-255 | 0-1 | -1..1 | ImageNet mean/std
  - image source : Dataset/Testing (full photo) and Dataset_cropped/Testing (hand crop)
and prints the accuracy of each. The combination that gives ~95%+ is the preprocessing the
model was trained with -> make the inference code use the same one.

Run from the repository root:
    python diagnose_cnn.py
    python diagnose_cnn.py --per-class 50
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

MODEL = "model/custom_cnn.tflite"
CLASS_MAP = "models/static/class_mapping.json"
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def parse_classes(obj):
    """Accepts list / {idx: label} / {label: idx} / nested dicts."""
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        if obj and all(str(k).isdigit() for k in obj):
            return [str(obj[k]) for k in sorted(obj, key=lambda k: int(k))]
        if obj and all(isinstance(v, int) for v in obj.values()):
            inv = {v: k for k, v in obj.items()}
            return [str(inv[i]) for i in sorted(inv)]
        for v in obj.values():
            if isinstance(v, (list, dict)):
                r = parse_classes(v)
                if r:
                    return r
    return None


def class_dirs(path: Path):
    return {d.name: d for d in sorted(path.iterdir()) if d.is_dir() and len(d.name) == 1} if path.is_dir() else {}


def prep(img_bgr, rgb, scale, size):
    x = cv2.resize(img_bgr, (size[1], size[0]))
    if rgb:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.astype(np.float32)
    if scale == "0-1":
        x /= 255.0
    elif scale == "-1..1":
        x = x / 127.5 - 1.0
    elif scale == "imagenet":
        x = (x / 255.0 - MEAN) / STD
    return x[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20, help="images sampled per letter (default 20)")
    ap.add_argument("--model", default=MODEL, help="tflite model to test (default model/custom_cnn.tflite)")
    args = ap.parse_args()
    random.seed(0)

    interp = Interpreter(model_path=args.model)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    size = (int(inp["shape"][1]), int(inp["shape"][2]))
    print(f"Model input {tuple(inp['shape'])} dtype={inp['dtype'].__name__}  output {tuple(out['shape'])}")

    classes = None
    try:
        classes = parse_classes(json.load(open(CLASS_MAP, encoding="utf-8")))
    except Exception as e:
        print(f"(could not read {CLASS_MAP}: {e})")
    n_out = int(out["shape"][-1])
    if not classes or len(classes) != n_out:
        classes = [chr(65 + i) for i in range(n_out)]
        print("Using A-Z as class order (class_mapping.json not usable).")
    print("Class order:", "".join(c[0] for c in classes), "\n")

    sources = {}
    for name, cands in {"Dataset/Testing": ["Dataset/Testing"],
                        "Dataset_cropped/Testing": ["Dataset_cropped/Testing", "Dataset_cropped"]}.items():
        for c in cands:
            d = class_dirs(Path(c))
            if d:
                sources[name] = d
                break
    if not sources:
        raise SystemExit("No Dataset/Testing/<A-Z> folders found. Run this from the repository root.")

    results = []  # (acc, source, color, scale, per_class_acc, confusions)
    for sname, dirs in sources.items():
        items = []
        for letter, d in dirs.items():
            files = [f for f in d.iterdir() if f.suffix.lower() in EXTS]
            for f in random.sample(files, min(args.per_class, len(files))):
                img = cv2.imread(str(f))
                if img is not None:
                    items.append((letter, img))
        print(f"{sname}: {len(items)} images")

        scales = ["raw"] if inp["dtype"] != np.float32 else ["raw", "0-1", "-1..1", "imagenet"]
        for rgb in (False, True):
            for scale in scales:
                ok, per, conf = Counter(), Counter(), Counter()
                for letter, img in items:
                    x = prep(img, rgb, scale, size).astype(inp["dtype"])
                    interp.set_tensor(inp["index"], x)
                    interp.invoke()
                    pred = classes[int(np.argmax(interp.get_tensor(out["index"])[0]))]
                    per[letter] += 1
                    if pred == letter:
                        ok[letter] += 1
                    else:
                        conf[(letter, pred)] += 1
                acc = sum(ok.values()) / max(1, sum(per.values()))
                results.append((acc, sname, "RGB" if rgb else "BGR", scale,
                                {l: ok[l] / per[l] for l in per}, conf))

    results.sort(key=lambda r: -r[0])
    print("\n=== Accuracy by preprocessing (best first) ===")
    for acc, s, col, sc, _, _ in results:
        print(f"  {acc * 100:5.1f}%   {s:<26} {col}  scale={sc}")

    acc, s, col, sc, per, conf = results[0]
    print(f"\nBest: {s}, {col}, scale={sc}  ->  {acc * 100:.1f}%")
    weak = sorted(per.items(), key=lambda kv: kv[1])[:8]
    print("Weakest letters :", ", ".join(f"{l} {a * 100:.0f}%" for l, a in weak))
    print("Top confusions  :", ", ".join(f"{a}->{b} x{n}" for (a, b), n in conf.most_common(8)))

    print("\nHow to read this:")
    print(" * Best accuracy ~95%+ -> the model is fine; make the app use that exact colour/scale/source.")
    print(" * Dataset_cropped >> Dataset -> the model expects a tight hand crop; crop the same way at inference.")
    print(" * Everything is low (<70%) -> the shipped model does not match this dataset; it must be retrained.")


if __name__ == "__main__":
    main()