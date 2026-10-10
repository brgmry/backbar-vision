"""Draw detection boxes over each photo for eyeballing. Usage: python src/draw.py <out_dir>"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener

register_heif_opener()

boxes = defaultdict(list)
for r in csv.DictReader(open("data/crops/boxes.csv")):
    boxes[r["photo"]].append(r)
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for photo, rows in boxes.items():
    img = Image.open(f"data/raw/{photo}.HEIC").convert("RGB")
    img.thumbnail((2400, 2400))
    d = ImageDraw.Draw(img)
    for r in rows:
        d.rectangle([float(r[k]) for k in ("x1", "y1", "x2", "y2")], outline="red", width=6)
        d.text((float(r["x1"]) + 4, float(r["y1"]) + 4), f"{r['crop']} {r['score']}", fill="yellow")
    img.save(out / f"{photo}.jpg", quality=85)
