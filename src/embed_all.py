"""Embed every image in a CSV index with DINOv2 (CLS, L2-normalised, float16), with the model's own resize and
centre crop: the features the trained head (train_head.py) learns from and identify_trained.py compares against.

Usage: python src/embed_all.py data/train/index.csv data/train/feats.npy   (also writes feats_ok.npy)
"""
import argparse
import csv
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import backbones

BATCH, WORKERS = 128, 10
LOG_EVERY = 20  # batches


def main(index: str, out: str) -> None:
    with open(index, newline="") as f:
        paths = [r["path"] for r in csv.DictReader(f)]
    emb, prep = backbones.load("dinov2", pad=False, dtype=torch.float16)
    feats, oks = [], []
    t, n = time.time(), 0
    with torch.no_grad():
        for x, ok in DataLoader(backbones.Images(paths, prep), batch_size=BATCH, num_workers=WORKERS):
            feats.append(emb(x).cpu().numpy().astype(np.float16))
            oks.append(ok.numpy())
            n += len(x)
            if n % (BATCH * LOG_EVERY) < BATCH:
                print(f"{n}/{len(paths)}  {n / (time.time() - t):.0f} img/s", flush=True)
    ok_mask = np.concatenate(oks)
    np.save(out, np.concatenate(feats))
    np.save(out.replace(".npy", "_ok.npy"), ok_mask)
    print(f"done: {n} images, {(~ok_mask).sum()} unreadable, {time.time() - t:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("index")
    ap.add_argument("out")
    args = ap.parse_args()
    main(args.index, args.out)
