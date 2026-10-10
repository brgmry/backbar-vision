"""Run fill-level estimators on top of any identification output. Alcohol only: other rows come out null.

The identification step is whatever wrote the predictions CSV (src/identify.py, src/match.py, ...), so
identifiers and fill estimators swap independently.

Usage: python src/experiment.py <predictions.csv> <fill spec> [<fill spec> ...]
  e.g. python src/experiment.py data/predictions_local_k3.csv geometry:edge geometry:opus vlm:opus
  -> data/predictions_fill_<ids>_<spec>.csv per spec (shared schema), overlays in runs/fill/<spec>/
Crops listed in data/labelled/crop_exclude.csv (image + bbox, checked by eye) are skipped: fill stays null.
Score: python src/score.py <that csv> --alcohol
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener

from fill import get_estimator

register_heif_opener()

ALCOHOL = {"spirit", "liqueur", "wine", "beer"}
PAD = 0.15  # detector boxes clip caps and bases; SAM gets the box, the crop gets room around it
EXCLUDE = Path("data/labelled/crop_exclude.csv")
PHOTO_DIRS = [Path("data/raw"), Path("data/pour")]
NULLED = ["description", "brand", "product", "category", "liquid", "size", "unit", "glass", "fill_ml",
          "id_source", "fill_visible"]


def padded_crop(photo: Image.Image, bbox: list[int]) -> tuple[Image.Image, list[float]]:
    x, y, w, h = bbox
    x0, y0 = max(0, round(x - PAD * w)), max(0, round(y - PAD * h))
    x1, y1 = min(photo.width, round(x + w + PAD * w)), min(photo.height, round(y + h + PAD * h))
    return photo.crop((x0, y0, x1, y1)), [x - x0, y - y0, x + w - x0, y + h - y0]


def overlay(crop: Image.Image, debug: dict, label: str) -> Image.Image:
    img = crop.convert("RGB")
    if "mask" in debug:
        tint = np.asarray(img).copy()
        tint[debug["mask"]] = (0.6 * tint[debug["mask"]] + [0, 100, 0]).astype(np.uint8)
        img = Image.fromarray(tint)
    draw = ImageDraw.Draw(img)
    for key, colour in (("full_y", "blue"), ("line_y", "red"), ("base_y", "cyan")):
        if debug.get(key) is not None:
            draw.line([(0, debug[key]), (img.width, debug[key])], fill=colour, width=2)
    if debug.get("elevation") is not None:
        label += f", camera {debug['elevation']:g} deg"
    draw.text((4, 4), label, fill="yellow")
    return img


def main(ids_path: str, specs: list[str]) -> None:
    rows = list(csv.DictReader(open(ids_path)))
    estimators = [get_estimator(s) for s in specs]
    excluded = {(r["image"], r["bbox"]) for r in csv.DictReader(open(EXCLUDE))} if EXCLUDE.exists() else set()
    results = {e.name: [] for e in estimators}
    photo_name, photo = None, None
    for row in rows:
        if row["category"] not in ALCOHOL:
            for e in estimators:
                results[e.name].append(row | {k: None for k in NULLED})
            continue
        if (row["image"], row["bbox"]) in excluded:
            for e in estimators:
                results[e.name].append(row | {"fill_ml": None, "fill_visible": False})
            continue
        if row["image"] != photo_name:
            photo_name, photo = row["image"], Image.open(next(d / row["image"] for d in PHOTO_DIRS if (d / row["image"]).exists())).convert("RGB")
        crop, box = padded_crop(photo, json.loads(row["bbox"]))
        for e in estimators:
            try:
                f = e.estimate(crop, box, row)
            except Exception as err:  # one bad bottle or API error shouldn't kill the run
                print(f"FAIL {e.name} {row['image']} {row['product']}: {err}", file=sys.stderr)
                results[e.name].append(row | {"fill_ml": None, "fill_visible": False})
                continue
            results[e.name].append(row | {"fill_ml": f.fill_ml, "fill_visible": f.visible})
            label = f"{row['product'][:18]}: {f.fill_ml if f.visible else 'not visible'}"
            out = Path("runs/fill") / e.name.replace(":", "_") / f"{Path(row['image']).stem}_{row['position']}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            overlay(crop, f.debug, label).save(out, quality=90)
            print(f"{e.name:<16} {row['image']} {row['product'][:30]:<30} {label.split(': ')[1]}")
    for name, out_rows in results.items():
        out = Path(f"data/predictions_fill_{Path(ids_path).stem.removeprefix('predictions_')}_{name.replace(':', '_')}.csv")
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"-> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
