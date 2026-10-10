"""Fuse three signals per crop to pick a brand, then the product within it:
  image  best catalogue-image similarity per brand (SigLIP2 image embeddings)
  name   SigLIP2 image-text similarity to "a photo of a bottle of <brand>"
  ocr    label text read by Apple Vision naming a brand (overrides when present)
Signals are z-scored per crop across brands and added with fixed, untuned weights.
Usage: python src/fuse.py <photos_dir> <crops_dir> <tag>   -> data/predictions_<tag>_siglip2_fuse.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from compare_backbones import embed_paths, full_res_crops, gallery_rows
from identify_trained import FIELDS, brand_names, read_brand

BB, W_IMAGE, W_NAME = "siglip2", 1.0, 1.0


def main(photos_dir, crops_dir, tag):
    rows = gallery_rows()
    G = torch.tensor(np.load(f"data/train/gal_{BB}.npy"))
    boxes, photos, paths = full_res_crops(photos_dir, crops_dir)
    Q = embed_paths(BB, paths, workers=0)
    texts = [r["text"] for r in csv.DictReader(open(Path(crops_dir) / "ocr_fullres.csv"))]
    names = brand_names(list(csv.DictReader(open("data/train/index.csv"))))

    # brand universe = catalogue brands; per-brand max image similarity
    brands = sorted({r["brand_key"] for r in rows if r["brand_key"]})
    bi = {b: k for k, b in enumerate(brands)}
    col = torch.tensor([bi.get(r["brand_key"], 0) for r in rows])
    img_sim = Q @ G.T
    per_brand = torch.full((len(Q), len(brands)), -1.0).scatter_reduce(1, col.expand(len(Q), -1), img_sim, "amax")

    # name similarity via the SigLIP2 text tower
    from backbones import BACKBONES, DEVICE
    model = AutoModel.from_pretrained(BACKBONES[BB]).to(DEVICE).eval()
    proc = AutoProcessor.from_pretrained(BACKBONES[BB])
    display = {r["brand_key"]: r["brand"] for r in rows}
    T = []
    with torch.no_grad():
        for i in range(0, len(brands), 256):
            batch = [f"a photo of a bottle of {display[b]}" for b in brands[i:i + 256]]
            t = proc(text=batch, return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(DEVICE)
            v = model.get_text_features(**t)
            T.append(F.normalize(getattr(v, "pooler_output", v).float(), dim=-1).cpu())
    name_sim = Q @ torch.cat(T).T

    z = lambda m: (m - m.mean(1, keepdim=True)) / (m.std(1, keepdim=True) + 1e-6)
    score = W_IMAGE * z(per_brand) + W_NAME * z(name_sim)
    by_brand = defaultdict(list)
    for j, r in enumerate(rows):
        by_brand[r["brand_key"]].append(j)

    best = defaultdict(dict)
    for k, b in enumerate(boxes):
        read = read_brand(texts[k], names)
        brand = read[0] if read and read[0] in bi else brands[int(score[k].argmax())]
        cand = by_brand[brand]
        j = cand[int(img_sim[k, cand].argmax())]
        s, r = float(score[k, bi[brand]]), rows[j]
        if s > best[b["photo"]].get(r["product_key"], (-1e9,))[0]:
            best[b["photo"]][r["product_key"]] = (s, k, r)

    out = []
    for photo, items in best.items():
        w, h = Image.open(photos[photo]).size
        sc = max(w, h) / min(max(w, h), 2400)
        for pos, (s, k, r) in enumerate(sorted(items.values(), key=lambda t: float(boxes[t[1]]["x1"])), 1):
            x1, y1, x2, y2 = (float(boxes[k][c]) * sc for c in ("x1", "y1", "x2", "y2"))
            out.append({"run": 1, "image": photos[photo].name, "position": pos, "description": f"crop {boxes[k]['crop']}, z {s:.2f}",
                        "bbox": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)], "brand": r["brand"], "product": r["product"],
                        "category": r["group"], "liquid": "", "size": r["size"], "unit": "", "glass": "", "fill_ml": None,
                        "id_source": "catalogue:siglip2:fuse", "fill_visible": False})
    path = f"data/predictions_{tag}_{BB}_fuse.csv"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(out)
    print(path)


if __name__ == "__main__":
    main(*sys.argv[1:4])
