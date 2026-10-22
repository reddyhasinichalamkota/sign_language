"""
Identify an ISL sign from an image and show the result in a UI window.

Usage (from the repository root):
    python src/predict_image.py                    # interactive: drag & drop / paste image paths
    python src/predict_image.py "Dataset/Testing/A/175.jpg"
    python src/predict_image.py path/to/image.jpg --no-ui      # print/save only

Pipeline: hand detection (MediaPipe) + static CNN -> label,
then hand segmentation (GrabCut / skin colour) -> hand shown in WHITE on black,
then a window with: Original | Hand segmentation | Detected sign.
Press any key in the window to close it. A copy is saved to evaluation_results/image_predictions/.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.inference.gesture_translator import ISLGestureTranslator
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger("PredictImage")

PANEL_H = 330
MAX_WINDOW_W = 1300
PLACEHOLDERS = ("Keep palm", "Analyzing", "Position hand", "Waiting")


def clean_label(label: str) -> str:
    """"Sign 'W' (WhatsApp)" -> "W (WhatsApp)"; "Sign 'A'" -> "A"."""
    if label.startswith("Sign '"):
        letter, _, rest = label[6:].partition("'")
        return f"{letter}{rest}".strip()
    return label


def skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    m1 = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m2 = cv2.inRange(hsv, (0, 20, 50), (25, 190, 255))
    return cv2.bitwise_and(m1, m2)


def segment_hand(img_bgr: np.ndarray, bbox) -> np.ndarray:
    """Binary uint8 mask (255 = hand). GrabCut inside the hand box, skin-colour as fallback."""
    h, w = img_bgr.shape[:2]
    if bbox is None:
        bbox = (0, 0, w, h)
    x1, y1, x2, y2 = bbox
    mask = np.zeros((h, w), np.uint8)

    # 1) GrabCut initialised with the hand box
    try:
        scale = min(1.0, 480.0 / max(h, w))
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
        rect = (int(x1 * scale), int(y1 * scale), max(2, int((x2 - x1) * scale)), max(2, int((y2 - y1) * scale)))
        gc = np.zeros(small.shape[:2], np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(small, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        mask = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
    except cv2.error:
        pass

    # 2) Skin-colour fallback if GrabCut gave nothing sensible (<3% or >90% of the box)
    box_area = max(1, (x2 - x1) * (y2 - y1))
    frac = cv2.countNonZero(mask[y1:y2, x1:x2]) / box_area
    if frac < 0.03 or frac > 0.90:
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = skin_mask(img_bgr[y1:y2, x1:x2])

    # Clean up: close holes, drop specks, keep the biggest blobs, smooth the edge
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = np.zeros_like(mask)
    big = [c for c in contours if cv2.contourArea(c) > 0.02 * box_area]
    cv2.drawContours(keep, big or contours, -1, 255, thickness=cv2.FILLED)
    return cv2.GaussianBlur(keep, (5, 5), 0)


def fit_height(img: np.ndarray, height: int = PANEL_H) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * height / h)), height))


def titled(panel: np.ndarray, title: str) -> np.ndarray:
    bar = np.full((34, panel.shape[1], 3), (30, 30, 35), np.uint8)
    cv2.putText(bar, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def predict_and_segment(image_path: str, translator):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image (not a valid image file?): {image_path}")

    translator.history.clear()  # no smoothing carry-over from a previous image
    annotated, label, conf, top3, bbox, _ = translator.process_frame(img.copy())

    # No usable result from the hand-tracking path: classify the whole image with the CNN
    if any(label.startswith(p) for p in PLACEHOLDERS) and translator.static_cnn.is_loaded:
        c_lbl, c_conf, top3 = translator.static_cnn.predict(img)
        label, conf = f"Sign '{c_lbl}'", c_conf

    mask = segment_hand(img, bbox)
    return img, annotated, mask, bbox, clean_label(label), conf, top3


def build_ui(img, annotated, mask, bbox, label, conf, top3) -> np.ndarray:
    # Segmentation panel: hand WHITE, everything else black
    white_hand = np.zeros_like(img)
    white_hand[mask > 127] = (255, 255, 255)

    # Result panel: background dimmed, hand whitened/highlighted, box drawn
    dim = (img * 0.25).astype(np.uint8)
    m3 = (mask.astype(np.float32) / 255.0)[..., None]
    highlight = (dim * (1 - m3) + (img * 0.35 + 255 * 0.65) * m3).astype(np.uint8)
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(highlight, contours, -1, (0, 255, 120), 2)
    if bbox is not None:
        cv2.rectangle(highlight, bbox[:2], bbox[2:], (0, 220, 255), 2)

    panels = [titled(fit_height(img), "1. Original image"),
              titled(fit_height(white_hand), "2. Hand segmentation"),
              titled(fit_height(highlight), "3. Hand highlighted")]
    row = np.hstack(panels)

    # Result banner
    banner = np.full((110, row.shape[1], 3), (20, 20, 25), np.uint8)
    cv2.putText(banner, f"Detected sign:  {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
    color = (0, 255, 0) if conf >= 0.7 else (0, 230, 255) if conf >= 0.4 else (120, 180, 255)
    cv2.putText(banner, f"Confidence: {conf * 100:.1f}%", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if top3:
        t = "   |   ".join(f"{l} {p * 100:.0f}%" for l, p in top3[:3])
        cv2.putText(banner, f"Top 3: {t}", (row.shape[1] // 2, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([row, banner])


def clean_path(text: str) -> str:
    """Accept drag-and-drop / pasted paths: strips quotes, a leading '& ' and surrounding spaces."""
    t = text.strip()
    if t.startswith("&"):
        t = t[1:].strip()
    return t.strip("\"'").strip()


def process_image(path: str, translator, show: bool = True):
    p = Path(clean_path(path))
    p = p if p.exists() else resolve_path(str(p))
    if not p.is_file():
        print(f"  Image not found: {p}")
        return
    try:
        img, annotated, mask, bbox, label, conf, top3 = predict_and_segment(str(p), translator)
    except ValueError as e:
        print(f"  {e}")
        return

    print(f"\n  Image     : {p}\n  Predicted : {label}\n  Confidence: {conf * 100:.1f}%")
    if top3:
        print("  Top 3     : " + ", ".join(f"{l} ({q * 100:.0f}%)" for l, q in top3[:3]))

    ui = build_ui(img, annotated, mask, bbox, label, conf, top3)
    out_dir = resolve_path("evaluation_results/image_predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{p.stem}_result.png"
    cv2.imwrite(str(out_file), ui)
    print(f"  Saved UI  : {out_file}")

    if show:
        if ui.shape[1] > MAX_WINDOW_W:  # fit the screen
            s = MAX_WINDOW_W / ui.shape[1]
            ui = cv2.resize(ui, (MAX_WINDOW_W, int(ui.shape[0] * s)))
        win = "ISL Sign Recognition - press any key to close"
        cv2.imshow(win, ui)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Identify an ISL sign from an image and show a segmentation UI")
    ap.add_argument("image", nargs="?", help="image path; omit it to paste / drag-and-drop images interactively")
    ap.add_argument("--no-ui", action="store_true", help="do not open the window")
    args = ap.parse_args()

    print("Loading models (one moment)...")
    translator = ISLGestureTranslator()
    try:
        if args.image:
            process_image(args.image, translator, show=not args.no_ui)
            return
        print("\n=== ISL Sign Recognition ===")
        print("Drag & drop an image into this window (or paste its path) and press Enter.")
        print("Type  q  and press Enter to quit.\n")
        while True:
            try:
                text = input("Image path > ")
            except (EOFError, KeyboardInterrupt):
                break
            if not text.strip():
                continue
            if text.strip().lower() in ("q", "quit", "exit"):
                break
            process_image(text, translator, show=not args.no_ui)
            print()
    finally:
        translator.close()


if __name__ == "__main__":
    main()