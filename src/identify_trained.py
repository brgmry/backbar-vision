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
import argparse
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from catalogue_match import (
    brand_names,
    brand_only,
    full_res_crops,
    keep_best,
    ocr_text,
    read_brand,
    read_rows,
    write_predictions,
)
from embed import embed
from train_head import Head

MODES = ("raw", "head", "cls", "ocr")


def load_head() -> tuple[Head, list[str]]:
    ck = torch.load("data/train/head.pt")
    head = Head(len(ck["brands"])).eval()
    head.load_state_dict(ck["state"])
    return head, ck["brands"]


def main(mode: str, photos_dir: str, crops_dir: str, out: str) -> None:
    rows = read_rows("data/train/index.csv")
    feats = np.load("data/train/feats.npy").astype(np.float32)
    ok = np.load("data/train/feats_ok.npy")
    gal = [i for i, r in enumerate(rows) if r["gallery"] == "1" and ok[i]]
    G = torch.tensor(feats[gal])
    by_brand = defaultdict(list)  # brand key -> positions in gal
    for j, i in enumerate(gal):
        by_brand[rows[i]["brand_key"]].append(j)

    with tempfile.TemporaryDirectory() as tmp:
        boxes, photos, crop_paths = full_res_crops(photos_dir, crops_dir, Path(tmp))
        Q = embed(crop_paths).float()
        texts = [ocr_text(p) for p in crop_paths] if mode == "ocr" else None

    with torch.no_grad():
        if mode == "raw":
            sims = Q @ G.T
        elif mode == "cls":
            head, brands = load_head()
            brand_pred = [brands[k] for k in head(Q)[1].argmax(1).tolist()]
            sims = Q @ G.T  # the product within the predicted brand is picked by raw features
        else:
            head, _ = load_head()
            sims = head.embed(Q) @ head.embed(G).T
    names = brand_names(rows) if mode == "ocr" else {}

    def nearest(k: int, cand: list[int] | None = None) -> tuple[float, dict]:
        """Score and catalogue row of crop k's nearest gallery image, among `cand` (positions in gal) if given."""
        if cand is None:
            s, j = (float(v) for v in sims[k].max(0))
            return s, rows[gal[int(j)]]
        j = cand[int(sims[k, cand].argmax())]
        return float(sims[k, j]), rows[gal[j]]

    best = defaultdict(dict)
    for k, b in enumerate(boxes):
        brand = read_brand(texts[k], names) if mode == "ocr" else None
        if brand:
            cand = by_brand.get(brand[0])
            # brand known (top-brands list) but no catalogue product: report the brand alone
            s, r = nearest(k, cand) if cand else (1.0, brand_only(brand))
        elif mode == "cls":
            cand = by_brand.get(brand_pred[k])
            if not cand:
                continue
            s, r = nearest(k, cand)
        else:
            s, r = nearest(k)
        keep_best(best, b["photo"], s, k, r)

    n = write_predictions(out, boxes, photos, best, f"catalogue:{mode}")
    print(f"{mode}: {n} rows over {len(best)} photos -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("photos_dir")
    ap.add_argument("crops_dir")
    ap.add_argument("out")
    args = ap.parse_args()
    main(args.mode, args.photos_dir, args.crops_dir, args.out)
