"""Fuse three signals per crop to pick a brand, then the product within it:
  image  best catalogue-image similarity per brand (SigLIP2 image embeddings)
  name   SigLIP2 image-text similarity to "a photo of a bottle of <brand>"
  ocr    label text read by Apple Vision naming a brand (overrides when present)
Signals are z-scored per crop across brands and added with fixed, untuned weights.
The gallery follows GALLERY like compare_backbones.py, and its embeddings come from `compare_backbones.py embed`.
Usage: python src/experiments/fuse.py <photos_dir> <crops_dir> <tag>   -> data/predictions_<tag>_siglip2_fuse.csv
"""
import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared helpers
from backbones import BACKBONES  # noqa: E402
from catalogue_match import brand_names, full_res_crops, keep_best, ocr_texts, read_brand, write_predictions  # noqa: E402
from compare_backbones import embed_paths, gallery  # noqa: E402
from models import DEVICE  # noqa: E402
from schema import read_csv  # noqa: E402

BB = "siglip2"
W_IMAGE = 1.0
W_NAME = 1.0
TEXT_BATCH = 256


def zscore(m: torch.Tensor) -> torch.Tensor:
    """Per crop (row), across brands, so signals on different scales can be added."""
    return (m - m.mean(1, keepdim=True)) / (m.std(1, keepdim=True) + 1e-6)


def name_embeddings(brands: list[str], display: dict[str, str]) -> torch.Tensor:
    """SigLIP2 text embeddings of "a photo of a bottle of <brand>" for each brand key."""
    model = AutoModel.from_pretrained(BACKBONES[BB]).to(DEVICE).eval()
    proc = AutoProcessor.from_pretrained(BACKBONES[BB])
    T = []
    with torch.no_grad():
        for i in range(0, len(brands), TEXT_BATCH):
            batch = [f"a photo of a bottle of {display[b]}" for b in brands[i:i + TEXT_BATCH]]
            t = proc(text=batch, return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(DEVICE)
            v = model.get_text_features(**t)
            T.append(F.normalize(getattr(v, "pooler_output", v).float(), dim=-1).cpu())
    return torch.cat(T)


def main(photos_dir, crops_dir, tag):
    rows, G = gallery(BB)
    with tempfile.TemporaryDirectory() as tmp:
        boxes, photos, paths = full_res_crops(photos_dir, crops_dir, Path(tmp))
        Q, _ = embed_paths(BB, paths, workers=0)
        texts = ocr_texts(crops_dir, boxes, paths)
    names = brand_names(read_csv("data/train/index.csv"))

    # brand universe = catalogue brands; per-brand max image similarity
    brands = sorted({r["brand_key"] for r in rows if r["brand_key"]})
    bi = {b: k for k, b in enumerate(brands)}
    col = torch.tensor([bi.get(r["brand_key"], 0) for r in rows])
    img_sim = Q @ G.T
    per_brand = torch.full((len(Q), len(brands)), -1.0).scatter_reduce(1, col.expand(len(Q), -1), img_sim, "amax")
    name_sim = Q @ name_embeddings(brands, {r["brand_key"]: r["brand"] for r in rows}).T

    score = W_IMAGE * zscore(per_brand) + W_NAME * zscore(name_sim)
    by_brand = defaultdict(list)
    for j, r in enumerate(rows):
        by_brand[r["brand_key"]].append(j)

    best = defaultdict(dict)
    for k, b in enumerate(boxes):
        read = read_brand(texts[k], names)
        brand = read[0] if read and read[0] in bi else brands[int(score[k].argmax())]
        cand = by_brand[brand]
        j = cand[int(img_sim[k, cand].argmax())]
        keep_best(best, b["photo"], float(score[k, bi[brand]]), k, rows[j])

    path = f"data/predictions_{tag}_{BB}_fuse.csv"
    write_predictions(path, boxes, photos, best, "catalogue:siglip2:fuse", label="z")
    print(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("photos_dir")
    ap.add_argument("crops_dir")
    ap.add_argument("tag")
    args = ap.parse_args()
    main(args.photos_dir, args.crops_dir, args.tag)
