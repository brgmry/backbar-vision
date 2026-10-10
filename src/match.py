"""Step 2+3: match each detected crop to the nearest reference crop, leave-one-photo-out.

For each held-out photo, the gallery is built from the other photos only: the k largest crops of
each inventory item, plus every clutter crop (label 0) so the matcher learns what isn't stock.
Writes one row per identified item in the shared output schema, so score.py scores it like the baseline.

Usage: python src/match.py [k ...]   -> data/predictions_local_k<k>.csv
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener

from embed import embed

register_heif_opener()

MIN_SIM = 0.3  # below this the crop is treated as clutter
FIELDS = ["run", "image", "position", "description", "bbox", "brand", "product", "category", "liquid",
          "size", "unit", "glass", "fill_ml", "id_source", "fill_visible"]

inventory = {int(r["id"]): r for r in csv.DictReader(open("data/labelled/inventory.csv"))}
CROPS = Path(os.environ.get("CROPS", "data/crops"))  # data/crops_whole: whole-bottle re-cut (src/recut_crops.py)
boxes = {(b["photo"], b["crop"]): b for b in csv.DictReader(open(CROPS / "boxes.csv"))}
crops = [r | {"item_id": int(r["item_id"])} for r in csv.DictReader(open(CROPS / "labels.csv"))]
for c in crops:
    b = boxes[(c["photo"], c["crop"])]
    c["box"] = [float(b[k]) for k in ("x1", "y1", "x2", "y2")]
    c["area"] = (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1])
vecs = embed([CROPS / c["photo"] / f"{int(c['crop']):02d}.jpg" for c in crops])
photos = sorted({c["photo"] for c in crops})
scale = {p: Image.open(f"data/raw/{p}.HEIC").width / 2400 for p in photos}  # detect.py resized to 2400px


def gallery(exclude: str, k: int) -> list[int]:
    per_item = defaultdict(list)
    for i, c in enumerate(crops):
        if c["photo"] != exclude and c["item_id"] >= 0:
            per_item[c["item_id"]].append(i)
    keep = [i for i in per_item.pop(0, [])]
    for idx in per_item.values():
        keep += sorted(idx, key=lambda i: -crops[i]["area"])[:k]
    return keep


def run(k: int) -> None:
    rows, correct, total = [], 0, 0
    for photo in photos:
        g = gallery(photo, k)
        held = [i for i, c in enumerate(crops) if c["photo"] == photo]
        sims = vecs[held] @ vecs[g].T
        best = {}  # item_id -> (sim, crop index): one row per item, keep the strongest crop
        for row, i in enumerate(held):
            s, j = sims[row].max(dim=0)
            pred = crops[g[int(j)]]["item_id"] if float(s) >= MIN_SIM else 0
            if crops[i]["item_id"] >= 0:
                total += 1
                correct += pred == crops[i]["item_id"]
            if pred and float(s) > best.get(pred, (-1,))[0]:
                best[pred] = (float(s), i)
        for pos, (item, (s, i)) in enumerate(sorted(best.items(), key=lambda kv: crops[kv[1][1]]["box"][0]), 1):
            x1, y1, x2, y2 = [v * scale[photo] for v in crops[i]["box"]]
            inv = inventory[item]
            rows.append({"run": 1, "image": f"{photo}.HEIC", "position": pos, "description": f"crop {crops[i]['crop']}, sim {s:.2f}",
                         "bbox": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)], "brand": inv["brand"],
                         "product": inv["product"], "category": inv["category"], "liquid": inv["liquid"],
                         "size": inv["size"], "unit": inv["unit"], "glass": "", "fill_ml": None,
                         "id_source": "reference", "fill_visible": False})
    suffix = "" if CROPS == Path("data/crops") else f"_{CROPS.name.removeprefix('crops_')}"
    out = Path(f"data/predictions_local{suffix}_k{k}.csv")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"k={k}: crop accuracy {correct}/{total} = {correct / total:.3f} -> {out}")


if __name__ == "__main__":
    for k in [int(a) for a in sys.argv[1:]] or [1, 2, 3]:
        run(k)
