"""Image embedding backbones behind one interface: name -> (embed(list[PIL]) -> L2-normalised tensor, dim).

Every backbone sees the image padded to a white square first, so a tall bottle is never cropped
(DINOv2's default centre crop cuts the top and bottom off a bottle, often the label).
"""
import torch
import torch.nn.functional as F
from PIL import Image

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BACKBONES = {
    "dinov2": "facebook/dinov2-base",
    "dinov2l": "facebook/dinov2-large",
    "siglip2": "google/siglip2-base-patch16-384",
    "cat2real": "zhanganyi88/Cat2Real-DINOv3-384",
    "marqo": "hf-hub:Marqo/marqo-ecommerce-embeddings-B",
}


def pad_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    out = Image.new("RGB", (s, s), (255, 255, 255))
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


class Prep:
    """Picklable preprocessing (DataLoader workers need it): pad to a white square, then the model's transform."""

    def __init__(self, fn=None, proc=None, size=224):
        self.fn, self.proc, self.size = fn, proc, size

    def __call__(self, img):
        img = pad_square(img)
        if self.fn is not None:
            return self.fn(img)
        img = img.resize((self.size, self.size), Image.BICUBIC)
        return self.proc(images=img, return_tensors="pt", do_resize=False, do_center_crop=False)["pixel_values"][0]


def load(name: str):
    """Returns (embed_fn, preprocess_fn). preprocess_fn: PIL -> tensor (CPU, for DataLoader workers);
    embed_fn: batch tensor -> normalised embeddings."""
    if name == "marqo":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(BACKBONES[name])
        model = model.to(DEVICE).eval()
        return (lambda x: F.normalize(model.encode_image(x.to(DEVICE)).float(), dim=-1)), Prep(fn=preprocess)
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(BACKBONES[name])
    size = 384 if name in ("siglip2", "cat2real") else 224
    model = AutoModel.from_pretrained(BACKBONES[name], dtype=torch.float16).to(DEVICE).eval()

    def emb(x):
        x = x.to(DEVICE, torch.float16)
        if name == "siglip2":
            v = model.get_image_features(pixel_values=x)
            v = getattr(v, "pooler_output", v)
        else:
            v = model(pixel_values=x).pooler_output
        return F.normalize(v.float(), dim=-1)
    return emb, Prep(proc=proc, size=size)
