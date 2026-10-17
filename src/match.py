"""Step 2+3: match each detected crop to the nearest reference crop, leave-one-photo-out.

For each held-out photo, the gallery is built from the other photos only: the k largest crops of
each inventory item, plus every clutter crop (label 0) so the matcher learns what isn't stock.
Writes one row per identified item in the shared output schema, so score.py scores it like the baseline.

Usage: python src/match.py [k ...]   -> data/predictions_local_k<k>.csv
"""
import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from PIL import Image

import photos  # noqa: F401  registers the HEIC opener
from boxes import area_xyxy, resized_scale
from detect import MAX_SIDE
from embed import embed
from identify import PRED_FIELDS

ROOT = Path(__file__).resolve().parent.parent
MIN_SIM = 0.3  # below this the crop is treated as clutter
CROPS = Path(os.environ.get("CROPS", ROOT / "data/crops"))  # data/crops_whole: whole-bottle re-cut (src/recut_crops.py)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_crops() -> list[dict]:
    """Every labelled crop with its box (in the photo shrunk by detect.py) and box area."""
    boxes = {(b["photo"], b["crop"]): b for b in read_csv(CROPS / "boxes.csv")}
    crops = []
    for r in read_csv(CROPS / "labels.csv"):
        b = boxes[(r["photo"], r["crop"])]
        box = [float(b[k]) for k in ("x1", "y1", "x2", "y2")]
        crops.append(r | {"item_id": int(r["item_id"]), "box": box, "area": area_xyxy(box)})
    return crops


def gallery(crops: list[dict], exclude: str, k: int) -> list[int]:
    per_item = defaultdict(list)
    for i, c in enumerate(crops):
        if c["photo"] != exclude and c["item_id"] >= 0:
            per_item[c["item_id"]].append(i)
    keep = per_item.pop(0, [])
    for idx in per_item.values():
        keep += sorted(idx, key=lambda i: -crops[i]["area"])[:k]
    return keep


def run(k: int, crops: list[dict], vecs, inventory: dict[int, dict], scale: dict[str, float]) -> None:
    rows, correct, total = [], 0, 0
    for photo in sorted(scale):
        g = gallery(crops, photo, k)
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
            rows.append({"run": 1, "image": f"{photo}.HEIC", "position": pos,
                         "description": f"crop {crops[i]['crop']}, sim {s:.2f}",
                         "bbox": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)], "brand": inv["brand"],
                         "product": inv["product"], "category": inv["category"], "liquid": inv["liquid"],
                         "size": inv["size"], "unit": inv["unit"], "glass": "", "fill_ml": None,
                         "id_source": "reference", "fill_visible": False})
    suffix = "" if CROPS == ROOT / "data/crops" else f"_{CROPS.name.removeprefix('crops_')}"
    out = ROOT / f"data/predictions_local{suffix}_k{k}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"k={k}: crop accuracy {correct}/{total} = {correct / total:.3f} -> {out}")


def main(ks: list[int]) -> None:
    inventory = {int(r["id"]): r for r in read_csv(ROOT / "data/labelled/inventory.csv")}
    crops = load_crops()
    vecs = embed([CROPS / c["photo"] / f"{int(c['crop']):02d}.jpg" for c in crops])
    scale = {p: resized_scale(Image.open(ROOT / f"data/raw/{p}.HEIC").size, MAX_SIDE)
             for p in {c["photo"] for c in crops}}
    for k in ks:
        run(k, crops, vecs, inventory, scale)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("k", nargs="*", type=int, default=[1, 2, 3], help="reference crops per item")
    main(ap.parse_args().k)
