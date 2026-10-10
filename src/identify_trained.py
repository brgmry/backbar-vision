"""Identify detected bottle crops against the external catalogue. Photos in, nothing from ground truth.

Modes:
  raw   nearest catalogue image by frozen DINOv2 features (zero-shot, no training)
  head  nearest catalogue image in the trained head's space
  cls   brand from the trained classifier, exact product by raw nearest neighbour within that brand
  ocr   brand from label text (Apple Vision OCR) matched against catalogue + top-brand names, exact product
        by head nearest neighbour within that brand; no readable brand -> head nearest neighbour
Crops are re-cut from the full-resolution photo (the saved crops come from a 2400px copy).
Usage: python src/identify_trained.py <mode> <photos_dir> <crops_dir> <out.csv>
"""
import csv
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pillow_heif import register_heif_opener

from rapidfuzz import fuzz

from embed import embed
from train_head import Head
from train_index import key

register_heif_opener()
FIELDS = ["run", "image", "position", "description", "bbox", "brand", "product", "category", "liquid",
          "size", "unit", "glass", "fill_ml", "id_source", "fill_visible"]


def main(mode: str, photos_dir: str, crops_dir: str, out: str) -> None:
    rows = list(csv.DictReader(open("data/train/index.csv")))
    feats = np.load("data/train/feats.npy").astype(np.float32)
    ok = np.load("data/train/feats_ok.npy")
    gal = [i for i, r in enumerate(rows) if r["gallery"] == "1" and ok[i]]
    G = torch.tensor(feats[gal])

    boxes = list(csv.DictReader(open(f"{crops_dir}/boxes.csv")))
    photos = {p.stem: p for p in Path(photos_dir).iterdir()}
    tmp = Path(tempfile.mkdtemp())
    crop_paths, cache = [], {}
    for k, b in enumerate(boxes):
        if b["photo"] not in cache:
            cache = {b["photo"]: Image.open(photos[b["photo"]]).convert("RGB")}
        img = cache[b["photo"]]
        sc = max(img.size) / min(max(img.size), 2400)
        x1, y1, x2, y2 = (float(b[c]) * sc for c in ("x1", "y1", "x2", "y2"))
        pw, ph = (x2 - x1) * 0.05, (y2 - y1) * 0.05
        crop_paths.append(tmp / f"{k}.jpg")
        img.crop((max(0, x1 - pw), max(0, y1 - ph), min(img.width, x2 + pw), min(img.height, y2 + ph))).save(crop_paths[-1], quality=95)
    Q = embed(crop_paths).float()

    ck = torch.load("data/train/head.pt") if mode in ("head", "cls", "ocr") else None
    if ck:
        head = Head(len(ck["brands"])).eval()
        head.load_state_dict(ck["state"])
    with torch.no_grad():
        if mode == "raw":
            sims = Q @ G.T
        elif mode == "head":
            sims = head.embed(Q) @ head.embed(G).T
        elif mode == "ocr":
            sims = head.embed(Q) @ head.embed(G).T
            by_brand = defaultdict(list)
            for j, i in enumerate(gal):
                by_brand[rows[i]["brand_key"]].append(j)
            names = brand_names(rows)
            texts = [ocr_text(p) for p in crop_paths]
        else:
            brand_pred = [ck["brands"][k] for k in head(Q)[1].argmax(1).tolist()]
            raw = Q @ G.T
            by_brand = defaultdict(list)
            for j, i in enumerate(gal):
                by_brand[rows[i]["brand_key"]].append(j)

    best = defaultdict(dict)  # photo -> product key -> (sim, box index, gallery row)
    for k, b in enumerate(boxes):
        if mode == "ocr":
            brand = read_brand(texts[k], names)
            if brand:
                cand = by_brand.get(brand[0], [])
                if cand:
                    j = cand[int(sims[k, cand].argmax())]
                    s = float(sims[k, j])
                    r = rows[gal[j]]
                else:  # brand known (top-brands list) but no catalogue product: report the brand alone
                    s, r = 1.0, {"brand": brand[1], "product": "", "product_key": f"brand:{brand[0]}", "group": "", "size": ""}
            else:
                s, j = (float(v) for v in sims[k].max(0))
                r = rows[gal[int(j)]]
            if s > best[b["photo"]].get(r["product_key"], (-1,))[0]:
                best[b["photo"]][r["product_key"]] = (s, k, r)
            continue
        if mode == "cls":
            cand = by_brand.get(brand_pred[k], [])
            if not cand:
                continue
            j = cand[int(raw[k, cand].argmax())]
            s = float(raw[k, j])
        else:
            s, j = (float(v) for v in sims[k].max(0))
            j = int(j)
        r = rows[gal[j]]
        if s > best[b["photo"]].get(r["product_key"], (-1,))[0]:
            best[b["photo"]][r["product_key"]] = (s, k, r)

    out_rows = []
    for photo, items in best.items():
        src = photos[photo]
        w, h = Image.open(src).size
        scale = max(w, h) / min(max(w, h), 2400)  # detect.py worked on the photo resized to 2400px
        for pos, (s, k, r) in enumerate(sorted(items.values(), key=lambda t: float(boxes[t[1]]["x1"])), 1):
            x1, y1, x2, y2 = (float(boxes[k][c]) * scale for c in ("x1", "y1", "x2", "y2"))
            out_rows.append({"run": 1, "image": src.name, "position": pos, "description": f"crop {boxes[k]['crop']}, sim {s:.2f}",
                             "bbox": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)], "brand": r["brand"],
                             "product": r["product"], "category": r["group"], "liquid": "", "size": r["size"], "unit": "",
                             "glass": "", "fill_ml": None, "id_source": f"catalogue:{mode}", "fill_visible": False})
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"{mode}: {len(out_rows)} rows over {len(best)} photos -> {out}")


def brand_names(rows) -> dict[str, str]:
    """brand_key -> display name, from the catalogue and the top-brands list. Short keys are too ambiguous to read."""
    names = {r["brand_key"]: r["brand"] for r in rows if r["gallery"] == "1"}
    for r in csv.DictReader(open("data/external/brands.csv")):
        names.setdefault(key(r["brand"]), r["brand"])
    return {k: v for k, v in names.items() if len(k.replace(" ", "")) >= 5}


def ocr_text(path: Path) -> str:
    from ocr import read_text
    return key(" ".join(t for t, _ in read_text(path)))


def read_brand(text: str, names: dict[str, str]):
    """Longest brand name found in the label text (fuzzy, to survive OCR slips like BACARD or JOUNNIE)."""
    if len(text) < 4:
        return None
    hits = [(k, v) for k, v in names.items() if fuzz.partial_ratio(k, text) >= 90 and len(text) >= len(k) * 0.6]
    return max(hits, key=lambda kv: len(kv[0])) if hits else None


if __name__ == "__main__":
    main(*sys.argv[1:5])
