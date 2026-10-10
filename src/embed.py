"""DINOv2 image embeddings, L2-normalised."""
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL = "facebook/dinov2-base"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

_processor = AutoImageProcessor.from_pretrained(MODEL)
_model = AutoModel.from_pretrained(MODEL).to(DEVICE).eval()


def embed(paths: list[Path], batch: int = 32) -> torch.Tensor:
    out = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        inputs = _processor(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            cls = _model(**inputs).pooler_output
        out.append(torch.nn.functional.normalize(cls, dim=-1).cpu())
    return torch.cat(out)
