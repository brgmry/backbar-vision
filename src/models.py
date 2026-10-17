"""The torch device and the SAM checkpoints, each loaded once on first use."""
from functools import cache

import numpy as np
import torch
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SAM2 = "facebook/sam2.1-hiera-large"
SAM3 = "facebook/sam3"  # gated on Hugging Face: accept Meta's licence there, then `hf auth login`
SAM3_SCORE = 0.3  # instances SAM 3 is less sure of than this are dropped
SAM3_MASK = 0.5  # pixel probability that counts as inside a mask


@cache
def sam2():
    from transformers import Sam2Model, Sam2Processor
    return Sam2Processor.from_pretrained(SAM2), Sam2Model.from_pretrained(SAM2).to(DEVICE).eval()


@cache
def sam3():
    from transformers import Sam3Model, Sam3Processor
    return Sam3Processor.from_pretrained(SAM3), Sam3Model.from_pretrained(SAM3).to(DEVICE).eval()


def sam3_instances(img: Image.Image, text: str, side: int) -> tuple[list[np.ndarray], list[float], float]:
    """Every SAM 3 instance for a text prompt. The image is first shrunk so its long side is at most `side` (SAM 3
    works at 1008 px internally). Returns the masks at that shrunk size, their scores, and the shrink factor."""
    processor, model = sam3()
    scale = min(1.0, side / max(img.size))
    small = img.resize((round(img.width * scale), round(img.height * scale))) if scale < 1 else img
    inputs = processor(images=small, text=text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    res = processor.post_process_instance_segmentation(out, threshold=SAM3_SCORE, mask_threshold=SAM3_MASK,
                                                       target_sizes=[[small.height, small.width]])[0]
    return list(res["masks"].cpu().numpy().astype(bool)), res["scores"].cpu().tolist(), scale


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A bool mask resized to (width, height), nearest neighbour."""
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
