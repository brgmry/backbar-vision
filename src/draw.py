"""Draw detection boxes over each photo for eyeballing.

Usage: python src/draw.py <out_dir> [--photos data/raw] [--crops data/crops]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import ImageDraw

import detect
from photos import EXTS


def main(out_dir: str, photos_dir: str, crops_dir: str) -> None:
    boxes = defaultdict(list)
    with open(Path(crops_dir) / "boxes.csv", newline="") as f:
        for r in csv.DictReader(f):
            boxes[r["photo"]].append(r)
    photos = {p.stem: p for p in Path(photos_dir).iterdir() if p.suffix.lower() in EXTS}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for photo, rows in boxes.items():
        img = detect.load(photos[photo])  # boxes.csv is in this shrunk copy
        d = ImageDraw.Draw(img)
        for r in rows:
            d.rectangle([float(r[k]) for k in ("x1", "y1", "x2", "y2")], outline="red", width=6)
            d.text((float(r["x1"]) + 4, float(r["y1"]) + 4), f"{r['crop']} {r['score']}", fill="yellow")
        img.save(out / f"{photo}.jpg", quality=85)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--photos", default="data/raw")
    ap.add_argument("--crops", default="data/crops")
    args = ap.parse_args()
    main(args.out_dir, args.photos, args.crops)
