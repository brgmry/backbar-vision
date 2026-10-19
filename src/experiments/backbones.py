"""Image embedding backbones behind one interface: name -> (embed(batch) -> L2-normalised tensor, preprocess).

Every backbone sees the image padded to a white square first, so a tall bottle is never cropped
(DINOv2's default centre crop cuts the top and bottom off a bottle, often the label).
pad=False keeps the model's own resize and centre crop instead: data/train/feats.npy, the trained head and the
leave-one-photo-out matcher were built that way, so their numbers only hold with the same features.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared helpers
from models import DEVICE  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True  # some catalogue downloads are cut short; DataLoader workers import this too
BACKBONES = {
    "dinov2": "facebook/dinov2-base",
    "dinov2l": "facebook/dinov2-large",
    "siglip2": "google/siglip2-base-patch16-384",
    "cat2real": "zhanganyi88/Cat2Real-DINOv3-384",
    "marqo": "hf-hub:Marqo/marqo-ecommerce-embeddings-B",
}
READ_ERRORS = (OSError, ValueError, Image.DecompressionBombError)  # a missing, corrupt or huge image file


def pad_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), (255, 255, 255))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


class Prep:
    """Picklable preprocessing (DataLoader workers need it): pad to a white square, then the model's transform."""

    def __init__(self, fn=None, proc=None, size=224, pad=True):
        self.fn, self.proc, self.size, self.pad = fn, proc, size, pad

    def __call__(self, img):
        if not self.pad:
            return self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        img = pad_square(img)
        if self.fn is not None:
            return self.fn(img)
        img = img.resize((self.size, self.size), Image.BICUBIC)
        return self.proc(images=img, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"][0]


class Images(Dataset):
    """Image files for a DataLoader, as (tensor, ok). A file that can't be read comes back as a blank image with
    ok False, so one bad download never stops a long run and the caller can drop it."""

    def __init__(self, paths, prep):
        self.paths, self.prep = paths, prep

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                return self.prep(im.convert("RGB")), True
        except READ_ERRORS:
            return self.prep(Image.new("RGB", (64, 64), (255, 255, 255))), False


def load(name: str, pad: bool = True, dtype: torch.dtype | None = None):
    """Returns (embed_fn, preprocess_fn). preprocess_fn: PIL -> tensor (CPU, for DataLoader workers);
    embed_fn: batch tensor -> normalised embeddings. dtype defaults to float16 (float32 for cat2real)."""
    if name == "marqo":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(BACKBONES[name])
        model = model.to(DEVICE).eval()
        return (lambda x: F.normalize(model.encode_image(x.to(DEVICE)).float(), dim=-1)), Prep(fn=preprocess)
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(BACKBONES[name])
    size = 384 if name in ("siglip2", "cat2real") else 224
    dtype = dtype or (torch.float32 if name == "cat2real" else torch.float16)  # ViT-L overflows to NaN in fp16
    model = AutoModel.from_pretrained(BACKBONES[name], dtype=dtype).to(DEVICE).eval()

    def emb(x):
        x = x.to(DEVICE, dtype)
        if name == "siglip2":
            v = model.get_image_features(pixel_values=x)
            v = getattr(v, "pooler_output", v)
        else:
            v = model(pixel_values=x).pooler_output
        return F.normalize(v.float(), dim=-1)
    return emb, Prep(proc=proc, size=size, pad=pad)
