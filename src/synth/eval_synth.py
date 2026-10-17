"""Does rendering TTB labels help? Real verified photos of the rendered brands are matched against four galleries.

Queries: OCR-verified real-world photos (data/external/wild/verified.csv) of the brands in data/synth/labels.csv,
both the whole photo and the most confident OWLv2 bottle crop. Never used to build any gallery.
Galleries: catalogue images (OFF, retailers, etc.), flat TTB labels, renders, catalogue + renders.
Score: top-1 brand among the rendered brands (max cosine per brand), SigLIP2.

Usage: python src/synth/eval_synth.py crops   (once: OWLv2 crops of the queries)
       python src/synth/eval_synth.py         -> table, appended to data/synth/results.txt
"""
import sys
from functools import cache
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, run as a script from the repo root
from boxes import padded_crop  # noqa: E402
from catalogue_match import CROP_PAD  # noqa: E402
from compare_backbones import embed_paths  # noqa: E402
from models import DEVICE  # noqa: E402
from photos import open_photo  # noqa: E402
from schema import read_csv  # noqa: E402

CROPS = Path("data/synth/query_crops")  # <slug>/<n>.jpg, mirroring data/external/wild
WILD = Path("data/external/wild")
QUERY_THRESHOLD = 0.1  # OWLv2 score for "a bottle"; the best box is kept, so a low bar only matters when unsure


@cache
def brands() -> list[str]:
    return sorted({r["brand_key"] for r in read_csv("data/synth/labels.csv")})


def queries() -> list[tuple[str, str]]:
    """(file under data/external/wild, brand key) of every verified photo of a rendered brand still on disk."""
    return [(r["file"], r["brand_key"]) for r in read_csv(WILD / "verified.csv")
            if r["brand_key"] in brands() and (WILD / r["file"]).is_file()]


def make_crops():
    from detect import load_model
    model, proc = load_model()
    for k, (file, _) in enumerate(queries()):
        out = CROPS / file  # the wild path, so photo <n> of two brands never share a crop
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        img = open_photo(WILD / file)
        inp = proc(text=[["a bottle"]], images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model(**inp)
        s = max(img.size)  # OWLv2 pads to a square
        r = proc.post_process_grounded_object_detection(o, threshold=QUERY_THRESHOLD,
                                                        target_sizes=torch.tensor([[s, s]]))[0]
        if len(r["scores"]):
            x1, y1, x2, y2 = r["boxes"][int(r["scores"].argmax())].tolist()
            img = padded_crop(img, [x1, y1, x2 - x1, y2 - y1], CROP_PAD)[0]
        img.save(out, quality=92)
        if k % 200 == 0:
            print(k, flush=True)


def galleries():
    idx = read_csv("data/train/index.csv")
    cat = [(r["path"], r["brand_key"]) for r in idx if r["brand_key"] in brands() and r["gallery"] == "1"
           and not r["source"].startswith("wild") and r["source"] != "ttb"]
    flat = [(r["label"], r["brand_key"]) for r in read_csv("data/synth/labels.csv")]
    renders = []
    for p in sorted(Path("data/synth/renders").glob("*.jpg")):
        b = p.stem.split("__")[0].replace("_", " ")
        if b in brands():
            renders.append((str(p), b))
    return {"catalogue": cat, "flat TTB labels": flat, "renders": renders, "catalogue + renders": cat + renders}


def top1(Q, qb, G, gb):
    sims = Q @ G.T
    keys = sorted(set(gb))
    per = torch.stack([sims[:, [i for i, b in enumerate(gb) if b == k]].max(1).values for k in keys], 1)
    pred = [keys[i] for i in per.argmax(1).tolist()]
    return np.array([p == t for p, t in zip(pred, qb, strict=True)])


def main():
    q = queries()
    qb = [b for _, b in q]
    missing = [f for f, _ in q if not (CROPS / f).is_file()]
    if missing:
        sys.exit(f"{len(missing)} query crops missing (e.g. {CROPS / missing[0]}): run `{sys.argv[0]} crops` first")
    Qs = {"whole photo": embed_paths("siglip2", [str(WILD / f) for f, _ in q])[0],
          "bottle crop": embed_paths("siglip2", [str(CROPS / f) for f, _ in q])[0]}
    gals = galleries()
    lines = [f"{len(q)} real photos, {len(brands())} brands, chance {1 / len(brands()):.3f}"]
    covered = {b for _, b in gals["catalogue"]}
    lines.append(f"catalogue covers {len(covered)}/{len(brands())} brands; renders {len(gals['renders'])} images")
    lines.append(f"{'gallery':22} {'n':>6} | {'whole: all':>10} {'covered':>8} | {'crop: all':>10} {'covered':>8}")
    cov = np.array([b in covered for b in qb])
    for name, g in gals.items():
        if not g:
            continue
        G = embed_paths("siglip2", [p for p, _ in g])[0]
        gb = [b for _, b in g]
        row = f"{name:22} {len(g):>6} |"
        for Q in Qs.values():
            ok = top1(Q, qb, G, gb)
            row += f" {ok.mean():>10.3f} {ok[cov].mean():>8.3f} |"
        lines.append(row)
    print("\n".join(lines))
    with open("data/synth/results.txt", "a") as f:
        f.write("\n".join(lines) + "\n\n")


if __name__ == "__main__":
    make_crops() if sys.argv[1:] == ["crops"] else main()
