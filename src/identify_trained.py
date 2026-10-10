"""Identify detected bottle crops against the external catalogue. Photos in, nothing from ground truth.

Modes:
  raw   nearest catalogue image by frozen DINOv2 features (zero-shot, no training)
  head  nearest catalogue image in the trained head's space
  cls   brand from the trained classifier, exact product by raw nearest neighbour within that brand
Usage: python src/identify_trained.py <mode> <photos_dir> <crops_dir> <out.csv>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pillow_heif import register_heif_opener

from embed import embed
from train_head import Head

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
    Q = embed([Path(f"{crops_dir}/{b['photo']}/{int(b['crop']):02d}.jpg") for b in boxes]).float()

    ck = torch.load("data/train/head.pt") if mode in ("head", "cls") else None
    if ck:
        head = Head(len(ck["brands"])).eval()
        head.load_state_dict(ck["state"])
    with torch.no_grad():
        if mode == "raw":
            sims = Q @ G.T
        elif mode == "head":
            sims = head.embed(Q) @ head.embed(G).T
        else:
            brand_pred = [ck["brands"][k] for k in head(Q)[1].argmax(1).tolist()]
            raw = Q @ G.T
            by_brand = defaultdict(list)
            for j, i in enumerate(gal):
                by_brand[rows[i]["brand_key"]].append(j)

    photos = {p.stem: p for p in Path(photos_dir).iterdir()}
    best = defaultdict(dict)  # photo -> product key -> (sim, box index, gallery row)
    for k, b in enumerate(boxes):
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


if __name__ == "__main__":
    main(*sys.argv[1:5])
