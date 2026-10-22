"""
FAST retraining (about 20-30 min on CPU instead of hours).

Idea: run every image ONCE through the frozen MobileNetV2 to get a 1280-number feature vector,
then train only a small classifier on those vectors (takes seconds). No per-epoch image decoding.

Two stages, so the numbers you quote stay honest:
  Stage A  train on Training only      -> report accuracy on Validation and Testing (HELD-OUT, honest)
  Stage B  train on Training+Validation+Testing -> this is the model that gets exported, because it has
           seen every kind of image in the dataset (different sessions / signers / backgrounds).
           Its accuracy on Dataset images is NOT a held-out score - do not quote it as one.

Output: model/custom_cnn_new.tflite  (drop-in: float32 RGB 0-255, 224x224x3 -> 26 softmax, A..Z)

Run from the repository root:
    python train_fast.py
"""
import argparse
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CLASSES = [chr(65 + i) for i in range(26)]
IMG = (224, 224)


def extract(root: Path, split: str, extractor, batch: int):
    d = root / split
    if not d.is_dir():
        raise SystemExit(f"Folder not found: {d}")
    ds = keras.utils.image_dataset_from_directory(
        d, labels="inferred", label_mode="int", class_names=CLASSES,
        image_size=IMG, batch_size=batch, shuffle=False)
    n_batches = int(ds.cardinality().numpy())
    feats, labels = [], []
    t0 = time.time()
    for i, (xb, yb) in enumerate(ds):
        feats.append(extractor(xb, training=False).numpy())
        labels.append(yb.numpy())
        if (i + 1) % 20 == 0 or i + 1 == n_batches:
            el = time.time() - t0
            print(f"  {split}: batch {i + 1}/{n_batches}  ({el / 60:.1f} min elapsed)", flush=True)
    return np.concatenate(feats), np.concatenate(labels)


def make_head():
    return keras.Sequential([
        layers.Input(shape=(1280,)),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])


def report(name, model, X, y):
    p = np.argmax(model.predict(X, batch_size=512, verbose=0), 1)
    acc = float(np.mean(p == y))
    per = {c: float(np.mean(p[y == i] == i)) for i, c in enumerate(CLASSES) if np.any(y == i)}
    conf = Counter((CLASSES[a], CLASSES[b]) for a, b in zip(y, p) if a != b)
    weak = sorted(per.items(), key=lambda kv: kv[1])[:8]
    print(f"  {name}: {acc * 100:.2f}%")
    print("     weakest:", ", ".join(f"{c} {v * 100:.0f}%" for c, v in weak))
    print("     confusions:", ", ".join(f"{a}->{b} x{n}" for (a, b), n in conf.most_common(8)))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Dataset")
    ap.add_argument("--out", default="model/custom_cnn_new.tflite")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    args = ap.parse_args()
    root = Path(args.data)

    weights = None if args.weights.lower() == "none" else "imagenet"
    inp = keras.Input(shape=IMG + (3,), name="raw_pixels_0_255")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)          # 0..255 -> -1..1 INSIDE the model
    base = keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False,
                                          weights=weights, pooling="avg")
    base.trainable = False
    extractor = keras.Model(inp, base(x, training=False), name="feature_extractor")

    print("=== 1/3 Extracting features (one pass over all images) ===")
    Xtr, ytr = extract(root, "Training", extractor, args.batch)
    Xva, yva = extract(root, "Validation", extractor, args.batch)
    Xte, yte = extract(root, "Testing", extractor, args.batch)
    print(f"Features: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")

    print("\n=== 2/3 Stage A: train on Training only -> HELD-OUT accuracy ===")
    head = make_head()
    head.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    h = head.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=128, verbose=0,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                                          restore_best_weights=True)])
    best_epochs = int(np.argmax(h.history["val_accuracy"])) + 1
    report("Validation (held-out)", head, Xva, yva)
    report("Testing    (held-out)", head, Xte, yte)

    print(f"\n=== 3/3 Stage B: train on ALL splits ({best_epochs}+ epochs), export ===")
    Xall, yall = np.concatenate([Xtr, Xva, Xte]), np.concatenate([ytr, yva, yte])
    final = make_head()
    final.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    final.fit(Xall, yall, epochs=max(best_epochs, 10), batch_size=128, verbose=0)
    report("All data (NOT held-out - model has seen these)", final, Xall, yall)

    full = keras.Model(inp, final(extractor(inp)), name="ISL_static_fast")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tf.lite.TFLiteConverter.from_keras_model(full).convert())
    print(f"\nSaved {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    # Sanity check of the exported file on a sample of Testing images (RGB, as the app feeds it)
    interp = tf.lite.Interpreter(model_path=str(out))
    interp.allocate_tensors()
    i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]
    ds = keras.utils.image_dataset_from_directory(root / "Testing", labels="inferred", label_mode="int",
                                                  class_names=CLASSES, image_size=IMG, batch_size=64,
                                                  shuffle=True, seed=1)
    ok = n = 0
    for xb, yb in ds.take(6):
        for img, t in zip(xb.numpy(), yb.numpy()):
            interp.set_tensor(i_d["index"], img[None].astype(np.float32))
            interp.invoke()
            ok += int(np.argmax(interp.get_tensor(o_d["index"])[0]) == t)
            n += 1
    print(f"TFLite file check on {n} Testing images (seen in Stage B): {ok / n * 100:.1f}%")
    print("\nNext: python diagnose_cnn.py --model model\\custom_cnn_new.tflite")


if __name__ == "__main__":
    main()