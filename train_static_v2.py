"""
Retrain the static ISL letter CNN and export a drop-in TFLite model.

Run from the repository root:
    python train_static.py                      # defaults: Dataset/, 8+10 epochs
    python train_static.py --head-epochs 5 --finetune-epochs 8

What it does
  * Trains on Dataset/Training, monitors Dataset/Validation, finally tests on Dataset/Testing.
  * MobileNetV2 (ImageNet) backbone -> 26-way softmax. Class order = A..Z (same as class_mapping.json).
  * The model takes RAW pixels 0-255, float32, 224x224x3 (the scaling is INSIDE the model), which is
    exactly what your current model expects, so predict_image.py / the app need no change.
  * Augmentation (zoom, shift, rotation, brightness, contrast, random RGB<->BGR swap) so it does not
    care whether the app feeds RGB or BGR, or a full photo or a hand crop.
  * Writes model/custom_cnn_new.tflite. Your old model is NOT touched until you copy it yourself.
"""
import argparse
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

CLASSES = [chr(65 + i) for i in range(26)]
IMG = (224, 224)


def load_split(root: Path, name: str, batch: int, shuffle: bool):
    d = root / name
    if not d.is_dir():
        raise SystemExit(f"Folder not found: {d}")
    return keras.utils.image_dataset_from_directory(
        d, labels="inferred", label_mode="categorical", class_names=CLASSES,
        image_size=IMG, batch_size=batch, shuffle=shuffle, seed=42)


def build_model(weights):
    try:
        base = keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False, weights=weights)
    except Exception as e:  # no internet for the ImageNet download
        print(f"!! Could not load ImageNet weights ({e}). Training from scratch - accuracy will be lower.")
        base = keras.applications.MobileNetV2(input_shape=IMG + (3,), include_top=False, weights=None)
    base.trainable = False

    inp = keras.Input(shape=IMG + (3,), name="raw_pixels_0_255")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)          # 0..255 -> -1..1 INSIDE the model
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inp, out), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Dataset")
    ap.add_argument("--out", default="model/custom_cnn_new.tflite")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--head-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    args = ap.parse_args()

    root = Path(args.data)
    train = load_split(root, "Training", args.batch, True)
    val = load_split(root, "Validation", args.batch, False)
    test = load_split(root, "Testing", args.batch, False)

    augment = keras.Sequential([
        layers.RandomRotation(0.06),
        layers.RandomZoom((-0.25, 0.15)),
        layers.RandomTranslation(0.10, 0.10),
        layers.RandomBrightness(0.25, value_range=(0, 255)),
        layers.RandomContrast(0.25),
    ])

    def aug(x, y):
        x = augment(x, training=True)
        x = tf.clip_by_value(x, 0.0, 255.0)
        x = tf.cond(tf.random.uniform([]) < 0.5, lambda: x[..., ::-1], lambda: x)  # random RGB<->BGR
        return x, y

    train = train.map(aug, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    model, base = build_model(None if args.weights.lower() == "none" else "imagenet")
    cbs = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-6)]

    print("\n=== Phase 1: train the new classifier head ===")
    model.compile(keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.head_epochs, callbacks=cbs)

    if args.finetune_epochs > 0:
        print("\n=== Phase 2: fine-tune the top of the backbone ===")
        base.trainable = True
        for l in base.layers[:-40]:
            l.trainable = False
        model.compile(keras.optimizers.Adam(1e-4), "categorical_crossentropy", metrics=["accuracy"])
        model.fit(train, validation_data=val, epochs=args.finetune_epochs, callbacks=cbs)

    print("\n=== Keras evaluation on Dataset/Testing ===")
    loss, acc = model.evaluate(test, verbose=0)
    print(f"Test accuracy: {acc * 100:.2f}%   loss: {loss:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tflite = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    out.write_bytes(tflite)
    print(f"\nSaved TFLite model -> {out}  ({len(tflite) / 1e6:.1f} MB)")

    # Check the exported .tflite itself, with both colour orders
    interp = tf.lite.Interpreter(model_path=str(out))
    interp.allocate_tensors()
    i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]
    stats = {"RGB": [0, 0], "BGR": [0, 0]}
    per, conf = Counter(), Counter()
    ok_by = Counter()
    for xb, yb in test:
        xb, yt = xb.numpy(), np.argmax(yb.numpy(), 1)
        for img, t in zip(xb, yt):
            for name, arr in (("RGB", img), ("BGR", img[..., ::-1])):
                interp.set_tensor(i_d["index"], arr[None].astype(np.float32))
                interp.invoke()
                p = int(np.argmax(interp.get_tensor(o_d["index"])[0]))
                stats[name][0] += int(p == t)
                stats[name][1] += 1
                if name == "RGB":
                    per[CLASSES[t]] += 1
                    if p == t:
                        ok_by[CLASSES[t]] += 1
                    else:
                        conf[(CLASSES[t], CLASSES[p])] += 1
    print("\n=== TFLite check on Dataset/Testing ===")
    for k, (a, n) in stats.items():
        print(f"  {k} input: {a / n * 100:.2f}%  ({a}/{n})")
    weak = sorted(per, key=lambda c: ok_by[c] / per[c])[:6]
    print("  Weakest letters:", ", ".join(f"{c} {ok_by[c] / per[c] * 100:.0f}%" for c in weak))
    print("  Top confusions :", ", ".join(f"{a}->{b} x{n}" for (a, b), n in conf.most_common(6)))
    print("\nIf this looks good: back up model\\custom_cnn.tflite, then copy the new file over it.")


if __name__ == "__main__":
    main()