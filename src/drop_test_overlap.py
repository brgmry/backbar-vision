"""Remove wild training images within pHash distance 6 of any test-set hash.

usage: python src/drop_test_overlap.py   (reads data/test_bars/phashes.txt, rewrites wild/manifest.csv)
"""
import csv, json, re
from pathlib import Path

import imagehash

ROOT = Path(__file__).resolve().parents[1] / "data"
WILD = ROOT / "external" / "wild"
THRESH = 6

test = []
f = ROOT / "test_bars" / "phashes.txt"
if f.exists():
    for line in f.read_text().splitlines():
        m = re.search(r"\b[0-9a-f]{16}\b", line)
        if m: test.append(imagehash.hex_to_hash(m.group(0)))
print("test hashes", len(test))

dropped = 0
for m in sorted(WILD.glob("*/_manifest.json")):
    rows = json.load(open(m)); keep = []
    for r in rows:
        h = imagehash.hex_to_hash(r["phash"])
        if any(h - t <= THRESH for t in test):
            (WILD / r["file"]).unlink(missing_ok=True); dropped += 1
        else:
            keep.append(r)
    json.dump(keep, open(m, "w"))
print("dropped", dropped)

rows = []
for m in sorted(WILD.glob("*/_manifest.json")): rows += json.load(open(m))
with open(WILD / "manifest.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["brand", "slug", "file", "source", "url", "page", "licence", "w", "h", "phash"])
    w.writeheader(); w.writerows(rows)
print("total", len(rows))
