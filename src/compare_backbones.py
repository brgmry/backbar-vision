"""Compare embedding backbones on the same catalogue and the same detected crops.

gallery: one image per catalogue product (external data only). For each backbone:
  raw  nearest catalogue product by image
  ocr  brand from label text (if readable) then nearest product within that brand, else raw
Usage: python src/compare_backbones.py embed <backbone>           -> data/train/gal_<bb>.npy (+ gal_<bb>_ok.npy)
       python src/compare_backbones.py predict <backbone> <photos_dir> <crops_dir> <tag>
                                                                   -> data/predictions_<tag>_<bb>_{raw,ocr}.csv
"""
import argparse
import csv
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import backbones
from catalogue_match import (
    brand_names,
    brand_only,
    full_res_crops,
    keep_best,
    ocr_texts,
    read_brand,
    read_rows,
    write_predictions,
)
from train_index import key

# GALLERY=v2 adds the OCR-verified real-world brand photos (brand-level entries) to the catalogue gallery
V = os.environ.get("GALLERY", "v1")
GAL = Path("data/train/gallery.csv" if V == "v1" else f"data/train/gallery_{V}.csv")
NPY = "data/train/gal_{bb}.npy" if V == "v1" else f"data/train/gal_{V}_{{bb}}.npy"


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def gallery_rows():
    if V == "v2" and not GAL.exists():
        base = read_rows("data/train/gallery.csv")
        for r in read_rows("data/external/wild/verified.csv"):
            base.append({**dict.fromkeys(base[0], ""), "path": f"data/external/wild/{r['file']}", "source": "wild_verified",
                         "brand": r["brand"], "brand_key": key(r["brand"]), "product_key": f"wild:{key(r['brand'])}", "gallery": "1"})
        write_rows(GAL, base)
    if not GAL.exists():
        seen, out = set(), []
        for r in read_rows("data/train/index.csv"):
            if r["gallery"] == "1" and r["product_key"] not in seen:
                seen.add(r["product_key"])
                out.append(r)
        write_rows(GAL, out)
    return read_rows(GAL)


def embed_paths(bb, paths, workers=4) -> tuple[torch.Tensor, np.ndarray]:
    """Embeddings of the image files, and which of them could be read (the rest are blank images)."""
    emb, prep = backbones.load(bb)
    out, oks = [], []
    with torch.no_grad():
        for x, ok in DataLoader(backbones.Images(paths, prep), batch_size=64, num_workers=workers):
            out.append(emb(x).cpu())
            oks.append(ok.numpy())
    return torch.cat(out), np.concatenate(oks)


def gallery(bb) -> tuple[list[dict], torch.Tensor]:
    """Gallery rows and their embeddings, leaving out images that could not be read when they were embedded."""
    rows, G = gallery_rows(), torch.tensor(np.load(NPY.format(bb=bb)))
    if len(rows) != len(G):
        sys.exit(f"{GAL} has {len(rows)} rows but {NPY.format(bb=bb)} has {len(G)}: run `embed {bb}` again")
    ok_file = Path(NPY.format(bb=bb).replace(".npy", "_ok.npy"))
    if ok_file.exists():  # galleries embedded before the mask existed have none
        ok = np.load(ok_file)
        rows, G = [r for r, o in zip(rows, ok, strict=True) if o], G[torch.from_numpy(ok)]
    return rows, G


def predict(bb, photos_dir, crops_dir, tag):
    rows, G = gallery(bb)
    with tempfile.TemporaryDirectory() as tmp:
        boxes, photos, paths = full_res_crops(photos_dir, crops_dir, Path(tmp))
        Q, _ = embed_paths(bb, paths, workers=0)
        texts = ocr_texts(crops_dir, boxes, paths)
    sims = Q @ G.T
    names = brand_names(read_rows("data/train/index.csv"))
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
                s, r = 1.0, brand_only(brand)
            else:
                s, j = (float(v) for v in sims[k].max(0))
                r = rows[int(j)]
                if not r["product"]:  # a brand-level wild photo: name the brand's closest catalogue product
                    cat = [c for c in by_brand.get(r["brand_key"], []) if rows[c]["product"]]
                    if cat:
                        r = rows[cat[int(sims[k, cat].argmax())]]
            keep_best(best, b["photo"], s, k, r)
        path = f"data/predictions_{tag}_{bb}{'' if V == 'v1' else '_' + V}_{mode}.csv"
        write_predictions(path, boxes, photos, best, f"catalogue:{bb}:{mode}")
        print(path)


def embed_gallery(bb):
    rows = gallery_rows()
    G, ok = embed_paths(bb, [r["path"] for r in rows])
    np.save(NPY.format(bb=bb), G.numpy())
    np.save(NPY.format(bb=bb).replace(".npy", "_ok.npy"), ok)
    print(f"{bb}: {len(rows)} products embedded, {(~ok).sum()} unreadable (left out of predictions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed").add_argument("backbone", choices=backbones.BACKBONES)
    p = sub.add_parser("predict")
    p.add_argument("backbone", choices=backbones.BACKBONES)
    for name in ("photos_dir", "crops_dir", "tag"):
        p.add_argument(name)
    args = ap.parse_args()
    if args.cmd == "embed":
        embed_gallery(args.backbone)
    else:
        predict(args.backbone, args.photos_dir, args.crops_dir, args.tag)
