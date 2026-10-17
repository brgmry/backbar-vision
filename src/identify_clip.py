"""Local zero-shot baseline, no VLM: OWLv2 crops (detect.py) + SigLIP2 zero-shot over a generic product list.

The label list is src/products_generic.py (best-selling brands and kitchen staples, written without
looking at the inventory). Crops closest to a clutter prompt are dropped. Same output schema as identify.py.

Usage: python src/identify_clip.py [data/raw] [--model google/siglip2-so400m-patch14-384]
       -> data/predictions_clip_siglip2.csv
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

import detect
from boxes import resized_scale
from identify import PRED_FIELDS, Bottle
from photos import EXTS
from products_generic import CLUTTER, products

TEMPLATES = ["a photo of a bottle of {}.", "a product photo of {}, {}."]
MIN_BOX = 8  # px in the detector's copy: thinner boxes are slivers, not bottles


@torch.no_grad()
def text_embeddings(model, processor, texts: list[str]) -> torch.Tensor:
    embs = []
    for i in range(0, len(texts), 64):
        inp = processor(text=texts[i:i + 64], padding="max_length", max_length=64, truncation=True,
                        return_tensors="pt").to(detect.DEVICE)
        e = model.get_text_features(**inp)
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        embs.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.cat(embs)


@torch.no_grad()
def image_embeddings(model, processor, crops: list[Image.Image]) -> torch.Tensor:
    inp = processor(images=crops, return_tensors="pt").to(detect.DEVICE)
    e = model.get_image_features(**inp)
    e = e.pooler_output if hasattr(e, "pooler_output") else e
    return torch.nn.functional.normalize(e, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="data/raw")
    ap.add_argument("--model", default="google/siglip2-so400m-patch14-384")
    ap.add_argument("--out", default="data/predictions_clip_siglip2.csv")
    args = ap.parse_args()

    owl, owl_proc = detect.load_model()
    proc = AutoProcessor.from_pretrained(args.model)
    clip = AutoModel.from_pretrained(args.model).to(detect.DEVICE).eval()

    catalogue = products()
    names = [f"{r['brand']} {r['product']}".strip() for r in catalogue]
    prod_emb = torch.stack([
        text_embeddings(clip, proc, [TEMPLATES[0].format(n) for n in names]),
        text_embeddings(clip, proc, [TEMPLATES[1].format(n, r["liquid"]) for n, r in zip(names, catalogue, strict=True)]),
    ]).mean(0)
    prod_emb = torch.nn.functional.normalize(prod_emb, dim=-1)
    clutter_emb = text_embeddings(clip, proc, [f"a photo of {c}." for c in CLUTTER])

    images = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS)
    times = []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        writer.writeheader()
        for path in images:
            t0 = time.time()
            with Image.open(path) as full:
                scale = resized_scale(full.size, detect.MAX_SIDE)
            img = detect.load(path)
            dets = sorted(detect.detect(owl, owl_proc, img), key=lambda d: d[1][0])
            boxes = [[max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)] for _, (x1, y1, x2, y2) in dets]
            boxes = [b for b in boxes if b[2] - b[0] > MIN_BOX and b[3] - b[1] > MIN_BOX]
            if not boxes:
                continue
            emb = image_embeddings(clip, proc, [img.crop(b) for b in boxes])
            prod_sim, clutter_sim = emb @ prod_emb.T, emb @ clutter_emb.T
            pos = 0
            for b, ps, cs in zip(boxes, prod_sim, clutter_sim, strict=True):
                best = int(ps.argmax())
                if cs.max() > ps[best]:
                    continue
                pos += 1
                r = catalogue[best]
                x1, y1, x2, y2 = (round(v * scale) for v in b)
                bottle = Bottle(position=pos, description=f"OWLv2 crop, SigLIP2 sim {float(ps[best]):.3f}",
                                bbox=[x1, y1, x2 - x1, y2 - y1], brand=r["brand"], product=r["product"],
                                category=r["category"], liquid=r["liquid"], size=r["size"], unit="ml",
                                glass="clear", fill_ml=None, id_source="shape", fill_visible=False)
                writer.writerow({"run": 1, "image": path.name, **bottle.model_dump()})
            dt = time.time() - t0
            times.append(dt)
            print(f"{path.name}: {len(boxes)} crops, {pos} kept, {dt:.1f}s", flush=True)
    print(f"-> {args.out}; mean {sum(times) / len(times):.1f}s per photo (models loaded)")


if __name__ == "__main__":
    main()
