"""Does rendering TTB labels help? Real verified photos of the rendered brands are matched against four galleries.

Queries: OCR-verified real-world photos (data/external/wild/verified.csv) of the brands in data/synth/labels.csv,
both the whole photo and the most confident OWLv2 bottle crop. Never used to build any gallery.
Galleries: catalogue images (OFF, retailers, etc.), flat TTB labels, renders, catalogue + renders.
Score: top-1 brand among the rendered brands (max cosine per brand), SigLIP2.

Usage: python src/synth/eval_synth.py crops   (once: OWLv2 crops of the queries)
       python src/synth/eval_synth.py         -> table, appended to data/synth/results.txt
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from compare_backbones import embed_paths  # noqa: E402

LAB = list(csv.DictReader(open("data/synth/labels.csv")))
BRANDS = sorted({r["brand_key"] for r in LAB})
CROPS = Path("data/synth/query_crops")
WILD = Path("data/external/wild")


def queries():
    return [(WILD / r["file"], r["brand_key"]) for r in csv.DictReader(open(WILD / "verified.csv")) if r["brand_key"] in BRANDS]


def make_crops():
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    CROPS.mkdir(parents=True, exist_ok=True)
    for k, (p, _) in enumerate(queries()):
        out = CROPS / f"{p.stem}.jpg"
        if out.exists():
            continue
        img = Image.open(p).convert("RGB")
        inp = proc(text=[["a bottle"]], images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**inp)
        s = max(img.size)
        r = proc.post_process_grounded_object_detection(o, threshold=0.1, target_sizes=torch.tensor([[s, s]]))[0]
        if len(r["scores"]):
            x1, y1, x2, y2 = r["boxes"][int(r["scores"].argmax())].tolist()
            pw, ph = (x2 - x1) * 0.05, (y2 - y1) * 0.05
            img = img.crop((max(0, x1 - pw), max(0, y1 - ph), min(img.width, x2 + pw), min(img.height, y2 + ph)))
        img.save(out, quality=92)
        if k % 200 == 0:
            print(k, flush=True)


def galleries():
    idx = list(csv.DictReader(open("data/train/index.csv")))
    cat = [(r["path"], r["brand_key"]) for r in idx if r["brand_key"] in BRANDS and r["gallery"] == "1"
           and not r["source"].startswith("wild") and r["source"] != "ttb"]
    flat = [(r["label"], r["brand_key"]) for r in LAB]
    renders = []
    for p in sorted(Path("data/synth/renders").glob("*.jpg")):
        b = p.stem.split("__")[0].replace("_", " ")
        if b in BRANDS:
            renders.append((str(p), b))
    return {"catalogue": cat, "flat TTB labels": flat, "renders": renders, "catalogue + renders": cat + renders}


def top1(Q, qb, G, gb):
    sims = Q @ G.T
    keys = sorted(set(gb))
    per = torch.stack([sims[:, [i for i, b in enumerate(gb) if b == k]].max(1).values for k in keys], 1)
    pred = [keys[i] for i in per.argmax(1).tolist()]
    return np.array([p == t for p, t in zip(pred, qb)])


def main():
    q = queries()
    qb = [b for _, b in q]
    Qs = {"whole photo": embed_paths("siglip2", [str(p) for p, _ in q]),
          "bottle crop": embed_paths("siglip2", [str(CROPS / f"{p.stem}.jpg") for p, _ in q])}
    gals = galleries()
    cache = {}
    lines = [f"{len(q)} real photos, {len(BRANDS)} brands, chance {1 / len(BRANDS):.3f}"]
    covered = {b for _, b in gals["catalogue"]}
    lines.append(f"catalogue covers {len(covered)}/{len(BRANDS)} brands; renders {len(gals['renders'])} images")
    lines.append(f"{'gallery':22} {'n':>6} | {'whole: all':>10} {'covered':>8} | {'crop: all':>10} {'covered':>8}")
    for name, g in gals.items():
        if not g:
            continue
        G = cache.setdefault(name, embed_paths("siglip2", [p for p, _ in g]))
        gb = [b for _, b in g]
        row = f"{name:22} {len(g):>6} |"
        for qn, Q in Qs.items():
            ok = top1(Q, qb, G, gb)
            cov = np.array([b in covered for b in qb])
            row += f" {ok.mean():>10.3f} {ok[cov].mean():>8.3f} |"
        lines.append(row)
    # per brand, crop queries, catalogue vs catalogue + renders
    print("\n".join(lines))
    with open("data/synth/results.txt", "a") as f:
        f.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    make_crops() if sys.argv[1:] == ["crops"] else main()
