"""
Make the live app trust the CNN over the hand-geometry rules.

Problem: in src/inference/gesture_translator.py the "Smart Fusion" step lets the rule-based
classifier (Open Hand -> B, Fist -> A, Peace -> V, Point -> D ...) win whenever it fires with a
fixed confidence >= 0.85, even when the CNN is right.
Fix: if the CNN is at least 50% sure, use the CNN; the rules only apply when the CNN is unsure.

Run from the repository root (a backup is kept as gesture_translator.py.bak):
    python patch_translator.py
Undo:  copy src\\inference\\gesture_translator.py.bak src\\inference\\gesture_translator.py
"""
import re
import shutil
import sys
from pathlib import Path

TARGET = Path("src/inference/gesture_translator.py")
CNN_FIRST_CONF = "0.50"

# tolerant: any indentation (spaces/tabs), extra spaces, trailing spaces
PATTERN = re.compile(
    r"^([ \t]*)if[ \t]+geom_label[ \t]+is[ \t]+not[ \t]+None[ \t]+and[ \t]+geom_conf[ \t]*>=[ \t]*0\.85[ \t]*:[ \t]*$",
    re.M)


def read_text(raw: bytes):
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16"), "utf-16"
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    return raw.decode("utf-8", errors="replace"), "utf-8"


def main():
    if not TARGET.is_file():
        sys.exit(f"Not found: {TARGET}  (run this from the repository root)")
    raw = TARGET.read_bytes()
    text, enc = read_text(raw)
    eol = "\r\n" if "\r\n" in text else "\n"
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if "CNN first:" in text:
        sys.exit("Already patched - nothing to do.")

    matches = list(PATTERN.finditer(text))
    if len(matches) != 1:
        print(f"Expected exactly 1 fusion line, found {len(matches)}. Nothing was changed.")
        print("Lines in your file that mention geom_conf (please paste these to me):")
        for n, line in enumerate(text.split("\n"), 1):
            if "geom_conf" in line or "Smart Fusion" in line:
                print(f"  {n}: {line!r}")
        sys.exit(1)

    m = matches[0]
    ind = m.group(1)
    new = (f"{ind}if cnn_label is not None and cnn_conf >= {CNN_FIRST_CONF}:\n"
           f"{ind}    # CNN first: the fixed-confidence geometry rules must not override a sure CNN\n"
           f"{ind}    pred_label = f\"Sign '{{cnn_label}}'\"\n"
           f"{ind}    pred_conf = cnn_conf\n"
           f"{ind}elif geom_label is not None and geom_conf >= 0.85:")
    text = text[:m.start()] + new + text[m.end():]

    shutil.copyfile(TARGET, str(TARGET) + ".bak")
    TARGET.write_bytes(text.replace("\n", eol).encode(enc))
    print(f"Patched {TARGET}  (backup: {TARGET}.bak)")


if __name__ == "__main__":
    main()