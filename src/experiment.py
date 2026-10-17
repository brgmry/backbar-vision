"""Run fill-level estimators on any identification output. Alcohol only by default; other rows come out null.

The identification step is whatever wrote the predictions CSV (src/identify.py, src/match.py, ...), so
identifiers and fill estimators swap independently. Each crop's bottle outline is computed once and shared by the
crop check and every estimator.

Usage: python src/experiment.py <predictions.csv> <fill spec> [<fill spec> ...] [--all] [--qc SPEC]
  e.g. python src/experiment.py data/predictions_local_k3.csv geometry:edge geometry:opus vlm:opus
  -> data/predictions_fill_<ids>_<spec>.csv per spec (shared schema), overlays in runs/fill/<spec>/
Crops the check rejects keep fill null. The crops rejected by hand (data/labelled/crop_exclude.csv) are ground
truth: they only score the check (src/qc.py), never decide which crops get a fill.
Score: python src/score.py <that csv> --alcohol
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from boxes import padded_crop
from fill import get_estimator, segment
from photos import open_photo
from qc import get_qc
from schema import ALCOHOL
from vlm import ERRORS as VLM_ERRORS

ROOT = Path(__file__).resolve().parent.parent
PHOTO_DIRS = [ROOT / "data/raw", ROOT / "data/pour", ROOT / "data/reference"]
NULLED = ["description", "brand", "product", "category", "liquid", "size", "unit", "glass", "fill_ml",
          "id_source", "fill_visible"]


def find_photo(name: str) -> Path:
    for d in PHOTO_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} is in none of {', '.join(str(d) for d in PHOTO_DIRS)}")


def overlay(crop: Image.Image, debug: dict, label: str) -> Image.Image:
    """The crop with the outline in green, any liquid mask in red, and the 100%, liquid and base rows as lines."""
    img = crop.convert("RGB")
    if "mask" in debug:
        tint = np.asarray(img).copy()
        tint[debug["mask"]] = (0.6 * tint[debug["mask"]] + [0, 100, 0]).astype(np.uint8)
        if debug.get("liquid") is not None:
            tint[debug["liquid"]] = (0.5 * tint[debug["liquid"]] + [120, 0, 0]).astype(np.uint8)
        img = Image.fromarray(tint)
    draw = ImageDraw.Draw(img)
    for key, colour in (("full_y", "blue"), ("line_y", "red"), ("base_y", "cyan")):
        if debug.get(key) is not None:
            draw.line([(0, debug[key]), (img.width, debug[key])], fill=colour, width=2)
    if debug.get("elevation") is not None:
        label += f", camera {debug['elevation']:g} deg"
    draw.text((4, 4), label, fill="yellow")
    return img


def main(ids_path: str, specs: list[str], categories: set[str] | None = ALCOHOL, qc_spec: str = "geometry") -> None:
    with open(ids_path, newline="") as f:
        rows = list(csv.DictReader(f))
    estimators = [get_estimator(s) for s in specs]
    check = None if qc_spec == "none" else get_qc(qc_spec)
    results = {e.name: [] for e in estimators}
    photo_name, photo = None, None
    for row in rows:
        if categories and row["category"] not in categories:
            for e in estimators:
                results[e.name].append(row | dict.fromkeys(NULLED))
            continue
        if row["image"] != photo_name:
            photo_name, photo = row["image"], open_photo(find_photo(row["image"]))
        crop, box = padded_crop(photo, json.loads(row["bbox"]))
        mask = segment(crop, box)
        verdict = check.check(crop, box, row, None, mask) if check else None
        if verdict and not verdict.ok:
            print(f"{'crop check':<16} {row['image']} {row['product'][:30]:<30} rejected: {verdict.reason}")
            for e in estimators:
                results[e.name].append(row | {"fill_ml": None, "fill_visible": False})
            continue
        for e in estimators:
            try:
                f = e.estimate(crop, box, row, mask)
            except VLM_ERRORS as err:  # a failed call costs one bottle, not the run
                print(f"FAIL {e.name} {row['image']} {row['product']}: {err}", file=sys.stderr)
                results[e.name].append(row | {"fill_ml": None, "fill_visible": False})
                continue
            results[e.name].append(row | {"fill_ml": f.fill_ml, "fill_visible": f.visible})
            label = f"{row['product'][:18]}: {f.fill_ml if f.visible else 'not visible'}"
            out = ROOT / "runs/fill" / e.name.replace(":", "_") / f"{Path(row['image']).stem}_{row['position']}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            overlay(crop, f.debug, label).save(out, quality=90)
            print(f"{e.name:<16} {row['image']} {row['product'][:30]:<30} {label.split(': ')[1]}")
    for name, out_rows in results.items():
        stem = Path(ids_path).stem.removeprefix("predictions_")
        out = ROOT / f"data/predictions_fill_{stem}_{name.replace(':', '_')}.csv"
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions")
    ap.add_argument("specs", nargs="+", help="fill estimators, see src/fill.py")
    ap.add_argument("--all", action="store_true", help="every category, not just alcohol")
    ap.add_argument("--qc", default="geometry", help="crop check run first: geometry, vlm:<model> or none")
    args = ap.parse_args()
    main(args.predictions, args.specs, None if args.all else ALCOHOL, args.qc)
