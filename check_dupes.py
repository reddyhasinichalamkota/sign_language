import csv
from collections import defaultdict
from pathlib import Path
from src.dataset.dataset_validator import hash_file

groups = defaultdict(list)
for split in ("Training", "Validation", "Testing"):
    for p in Path("Dataset", split).glob("*/*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            groups[hash_file(p)].append((split, p.parent.name, str(p)))

dups = {h: v for h, v in groups.items() if len(v) > 1}
conflict = {h: v for h, v in dups.items() if len({l for _, l, _ in v}) > 1}
cross = {h: v for h, v in dups.items() if len({s for s, _, _ in v}) > 1}
cross_same = {h: v for h, v in cross.items() if h not in conflict}

print("Duplicate groups total      :", len(dups))
print("Same image, DIFFERENT letter:", len(conflict), "(label noise)")
print("Across splits, same letter  :", len(cross_same), "(leakage)")
print("\nExamples of different-letter duplicates:")
for v in list(conflict.values())[:8]:
    print("  ", [(s, l) for s, l, _ in v])

with open("dupes_report.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "split", "letter", "path"])
    for kind, d in (("conflict", conflict), ("leak", cross_same)):
        for v in d.values():
            for s, l, p in v:
                w.writerow([kind, s, l, p])
print("\nSaved dupes_report.csv")