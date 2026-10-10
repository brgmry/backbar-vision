"""Embed every image in a CSV index with DINOv2 (CLS, L2-normalised, float16).

Usage: python src/embed_all.py data/train/index.csv data/train/feats.npy
"""
import csv
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL = "facebook/dinov2-base"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL)


class Images(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            ok = True
        except Exception:
            img, ok = Image.new("RGB", (224, 224)), False
        return processor(images=img, return_tensors="pt")["pixel_values"][0], ok


def main(index: str, out: str) -> None:
    paths = [r["path"] for r in csv.DictReader(open(index))]
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    feats = np.zeros((len(paths), model.config.hidden_size), dtype=np.float16)
    ok_mask = np.zeros(len(paths), dtype=bool)
    loader = DataLoader(Images(paths), batch_size=128, num_workers=10)
    t, n = time.time(), 0
    with torch.no_grad():
        for x, ok in loader:
            v = model(pixel_values=x.to(DEVICE, torch.float16)).pooler_output.float()
            feats[n:n + len(x)] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
            ok_mask[n:n + len(x)] = ok.numpy()
            n += len(x)
            if n % (128 * 20) < 128:
                print(f"{n}/{len(paths)}  {n / (time.time() - t):.0f} img/s", flush=True)
    np.save(out, feats)
    np.save(out.replace(".npy", "_ok.npy"), ok_mask)
    print(f"done: {n} images, {(~ok_mask).sum()} unreadable, {time.time() - t:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
