"""OCR every detected bottle, cropping from the full-resolution original photo.

Usage: python src/ocr_crops.py   -> data/crops/ocr.csv (photo, crop, text)
"""
import csv
import tempfile
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

from ocr import read_text

register_heif_opener()
PAD = 0.05  # widen each box a little so edge letters are not cut

boxes = list(csv.DictReader(open("data/crops/boxes.csv")))
rows, cache = [], {}
with tempfile.TemporaryDirectory() as tmp:
    for b in boxes:
        if b["photo"] not in cache:
            cache = {b["photo"]: Image.open(f"data/raw/{b['photo']}.HEIC").convert("RGB")}
        img = cache[b["photo"]]
        s = img.width / 2400  # boxes.csv is in the 2400px-resized photo
        x1, y1, x2, y2 = (float(b[k]) * s for k in ("x1", "y1", "x2", "y2"))
        pw, ph = (x2 - x1) * PAD, (y2 - y1) * PAD
        crop = img.crop((max(0, x1 - pw), max(0, y1 - ph), min(img.width, x2 + pw), min(img.height, y2 + ph)))
        path = Path(tmp) / "c.png"
        crop.save(path)
        text = " | ".join(t for t, _ in read_text(path))
        rows.append({"photo": b["photo"], "crop": b["crop"], "text": text})
        print(f"{b['photo']} {b['crop']:>2}: {text}", flush=True)
with open("data/crops/ocr.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["photo", "crop", "text"])
    w.writeheader()
    w.writerows(rows)
