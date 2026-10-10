"""Compare embedding backbones on the same catalogue and the same detected crops.

gallery: one image per catalogue product (external data only). For each backbone:
  raw  nearest catalogue product by image
  ocr  brand from label text (if readable) then nearest product within that brand, else raw
Usage: python src/compare_backbones.py embed <backbone>           -> data/train/gal_<bb>.npy
       python src/compare_backbones.py predict <backbone> <photos_dir> <crops_dir> <tag>
                                                                   -> data/predictions_<tag>_<bb>_{raw,ocr}.csv
"""
import csv
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from pillow_heif import register_heif_opener
from torch.utils.data import DataLoader, Dataset

import backbones
from identify_trained import FIELDS, brand_names, read_brand
from ocr import read_text
from train_index import key

register_heif_opener()
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os

# GALLERY=v2 adds the OCR-verified real-world brand photos (brand-level entries) to the catalogue gallery
V = os.environ.get("GALLERY", "v1")
GAL = Path("data/train/gallery.csv" if V == "v1" else f"data/train/gallery_{V}.csv")
NPY = "data/train/gal_{bb}.npy" if V == "v1" else f"data/train/gal_{V}_{{bb}}.npy"


def gallery_rows():
    if V == "v2" and not GAL.exists():
        base = list(csv.DictReader(open("data/train/gallery.csv")))
        for r in csv.DictReader(open("data/external/wild/verified.csv")):
            base.append({**{k: "" for k in base[0]}, "path": f"data/external/wild/{r['file']}", "source": "wild_verified",
                         "brand": r["brand"], "brand_key": key(r["brand"]), "product_key": f"wild:{key(r['brand'])}", "gallery": "1"})
        with GAL.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(base[0]))
            w.writeheader()
            w.writerows(base)
    if not GAL.exists():
        seen, out = set(), []
        for r in csv.DictReader(open("data/train/index.csv")):
            if r["gallery"] == "1" and r["product_key"] not in seen:
                seen.add(r["product_key"])
                out.append(r)
        with GAL.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    return list(csv.DictReader(GAL.open()))


class Imgs(Dataset):
    def __init__(self, paths, prep):
        self.paths, self.prep = paths, prep

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.prep(Image.open(self.paths[i]).convert("RGB"))
        except Exception:
            return self.prep(Image.new("RGB", (64, 64), (255, 255, 255)))


def embed_paths(bb, paths, workers=4):
    emb, prep = backbones.load(bb)
    out = []
    with torch.no_grad():
        for x in DataLoader(Imgs(paths, prep), batch_size=64, num_workers=workers):
            out.append(emb(x).cpu())
    return torch.cat(out)


def full_res_crops(photos_dir, crops_dir):
    boxes = list(csv.DictReader(open(f"{crops_dir}/boxes.csv")))
    photos = {p.stem: p for p in Path(photos_dir).iterdir()}
    tmp, paths, cache = Path(tempfile.mkdtemp()), [], {}
    for k, b in enumerate(boxes):
        if b["photo"] not in cache:
            cache = {b["photo"]: Image.open(photos[b["photo"]]).convert("RGB")}
        img = cache[b["photo"]]
        sc = max(img.size) / min(max(img.size), 2400)
        x1, y1, x2, y2 = (float(b[c]) * sc for c in ("x1", "y1", "x2", "y2"))
        pw, ph = (x2 - x1) * 0.05, (y2 - y1) * 0.05
        paths.append(tmp / f"{k}.jpg")
        img.crop((max(0, x1 - pw), max(0, y1 - ph), min(img.width, x2 + pw), min(img.height, y2 + ph))).save(paths[-1], quality=95)
    return boxes, photos, paths


def predict(bb, photos_dir, crops_dir, tag):
    rows = gallery_rows()
    G = torch.tensor(np.load(NPY.format(bb=bb)))
    boxes, photos, paths = full_res_crops(photos_dir, crops_dir)
    Q = embed_paths(bb, paths, workers=0)
    sims = Q @ G.T
    ocr_cache = Path(crops_dir) / "ocr_fullres.csv"
    if ocr_cache.exists():
        texts = [r["text"] for r in csv.DictReader(ocr_cache.open())]
    else:
        texts = [key(" ".join(t for t, _ in read_text(p))) for p in paths]
        with ocr_cache.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text"])
            w.writerows([[t] for t in texts])
    names = brand_names(list(csv.DictReader(open("data/train/index.csv"))))
    by_brand = defaultdict(list)
    for j, r in enumerate(rows):
        by_brand[r["brand_key"]].append(j)
    for mode in ("raw", "ocr"):
        best = defaultdict(dict)
        for k, b in enumerate(boxes):
            brand = read_brand(texts[k], names) if mode == "ocr" else None
            if brand and by_brand.get(brand[0]):
                cand = by_brand[brand[0]]
                j = cand[int(sims[k, cand].argmax())]
                s, r = float(sims[k, j]), rows[j]
            elif brand:
                s, r = 1.0, {"brand": brand[1], "product": "", "product_key": f"brand:{brand[0]}", "group": "", "size": ""}
            else:
                s, j = (float(v) for v in sims[k].max(0))
                r = rows[int(j)]
                if not r["product"]:
                    cat = [c for c in by_brand.get(r["brand_key"], []) if rows[c]["product"]]
                    if cat:
                        r = rows[cat[int(sims[k, cat].argmax())]]
            if s > best[b["photo"]].get(r["product_key"], (-1,))[0]:
                best[b["photo"]][r["product_key"]] = (s, k, r)
        out = []
        for photo, items in best.items():
            w, h = Image.open(photos[photo]).size
            sc = max(w, h) / min(max(w, h), 2400)
            for pos, (s, k, r) in enumerate(sorted(items.values(), key=lambda t: float(boxes[t[1]]["x1"])), 1):
                x1, y1, x2, y2 = (float(boxes[k][c]) * sc for c in ("x1", "y1", "x2", "y2"))
                out.append({"run": 1, "image": photos[photo].name, "position": pos, "description": f"crop {boxes[k]['crop']}, sim {s:.2f}",
                            "bbox": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)], "brand": r["brand"], "product": r["product"],
                            "category": r["group"], "liquid": "", "size": r["size"], "unit": "", "glass": "", "fill_ml": None,
                            "id_source": f"catalogue:{bb}:{mode}", "fill_visible": False})
        path = f"data/predictions_{tag}_{bb}{'' if V == 'v1' else '_' + V}_{mode}.csv"
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(out)
        print(path)


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        bb = sys.argv[2]
        rows = gallery_rows()
        G = embed_paths(bb, [r["path"] for r in rows])
        np.save(NPY.format(bb=bb), G.numpy())
        print(f"{bb}: {len(rows)} products embedded")
    else:
        predict(*sys.argv[2:6])
