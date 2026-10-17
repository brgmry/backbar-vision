"""DINOv2 image embeddings, L2-normalised, for the leave-one-photo-out matcher (match.py) and the trained head
(identify_trained.py). The model loads on the first call, not on import.

These keep DINOv2's own resize and centre crop and run in float32, as they always have: the head was trained on
features made that way (embed_all.py) and the matcher's numbers were measured with them.
"""
from functools import cache
from pathlib import Path

import torch
from PIL import Image

import backbones


@cache
def _model():
    return backbones.load("dinov2", pad=False, dtype=torch.float32)


def embed(paths: list[Path], batch: int = 32) -> torch.Tensor:
    emb, prep = _model()
    out = []
    for i in range(0, len(paths), batch):
        x = []
        for p in paths[i:i + batch]:
            with Image.open(p) as im:
                x.append(prep(im.convert("RGB")))
        with torch.no_grad():
            out.append(emb(torch.stack(x)).cpu())
    return torch.cat(out)
