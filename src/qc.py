"""Crop check: reject crops that can't support a fill reading (cut off, hidden, the wrong thing) before fill runs.

Pick one with a spec string, like the fill estimators:
  geometry       local and free, from the bottle outline: it touches the crop's top or bottom (cut off), has no
                 neck, fills too little of its box, or its height/width ratio is far from the same item elsewhere
  vlm:<model>    asks the VLM whether the whole bottle is visible, cap to base, with nothing in front

Scored against the crops rejected by hand (data/labelled/crop_exclude.csv):
  python src/qc.py <predictions.csv> <qc spec> [<qc spec> ...]
"""
import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from pydantic import BaseModel

from boxes import padded_crop
from fill import describe, fit, neck_start, row_widths, segment
from photos import open_photo
from schema import ALCOHOL
from vlm import get_vlm

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = ROOT / "data/labelled/crop_exclude.csv"  # ground truth for scoring the check, never an input to it
EDGE_PX = 3  # an outline this close to the crop's top or bottom runs out of the crop
MIN_FILL = 0.55  # outline area / box area; below this the bottle is hidden or the box is wrong
MAX_SHAPE_DEV = 0.25  # height/width ratio allowed away from the item's median across photos
MIN_ROWS = 20  # an outline shorter than this is no bottle


class Verdict(BaseModel):
    ok: bool
    reason: str


class CropQC(Protocol):
    name: str

    def check(self, crop: Image.Image, box: list[float], item: dict, shape_ref: float | None,
              mask: np.ndarray | None = None) -> Verdict: ...


def height_ratio(mask: np.ndarray) -> float | None:
    """Outline height over its body width, or None when the outline is too small to be a bottle."""
    width = row_widths(mask)
    rows = np.flatnonzero(width)
    if rows.size < MIN_ROWS:
        return None
    body = np.median(width[(rows[0] + rows[-1]) // 2:rows[-1] + 1])
    return (rows[-1] - rows[0]) / body


class GeometryQC:
    name = "geometry"

    def check(self, crop, box, item, shape_ref, mask=None):
        mask = segment(crop, box) if mask is None else mask
        ratio = height_ratio(mask)
        if ratio is None:
            return Verdict(ok=False, reason="no bottle outline")
        rows = np.flatnonzero(mask.any(axis=1))
        if rows[0] < EDGE_PX or rows[-1] > mask.shape[0] - 1 - EDGE_PX:
            return Verdict(ok=False, reason="bottle runs out of the crop")
        if neck_start(row_widths(mask), rows[0], rows[-1]) <= rows[0]:
            return Verdict(ok=False, reason="no neck found")
        x1, y1, x2, y2 = box
        if mask.sum() / ((x2 - x1) * (y2 - y1)) < MIN_FILL:
            return Verdict(ok=False, reason="outline fills too little of the box")
        if shape_ref and abs(ratio / shape_ref - 1) > MAX_SHAPE_DEV:
            return Verdict(ok=False, reason=f"shape {ratio:.2f} vs usual {shape_ref:.2f}")
        return Verdict(ok=True, reason="")


class VLMQC:
    def __init__(self, model: str):
        self.vlm = get_vlm(model)
        self.name = f"vlm:{model}"

    def check(self, crop, box, item, shape_ref, mask=None):
        prompt = (f"This crop should show one bottle: {describe(item)}. ok: true only if that bottle is in the "
                  "middle of the crop and fully visible from cap to base, with nothing in front of the part that "
                  "holds liquid. reason: a few words when ok is false.")
        return self.vlm.ask([fit(crop)[0]], prompt, Verdict)


def get_qc(spec: str) -> CropQC:
    kind, _, arg = spec.partition(":")
    if kind == "geometry":
        return GeometryQC()
    if kind == "vlm":
        return VLMQC(arg)
    raise ValueError(f"unknown crop check {spec!r}: use geometry or vlm:<model>")


def shape_refs(rows: list[dict], masks: dict) -> dict[str, float]:
    """Median height/width per item across all its crops: a cut-off or hidden bottle stands out against it."""
    by_item = {}
    for r in rows:
        ratio = height_ratio(masks[(r["image"], r["bbox"])])
        if ratio:
            by_item.setdefault(r["product"], []).append(ratio)
    return {k: statistics.median(v) for k, v in by_item.items()}


def main(ids_path: str, specs: list[str]) -> None:
    with open(ids_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["category"] in ALCOHOL]
    with open(EXCLUDE, newline="") as f:
        bad = {(r["image"], r["bbox"]): r["reason"] for r in csv.DictReader(f)}
    photos = {img: open_photo(ROOT / "data/raw" / img) for img in {r["image"] for r in rows}}
    crops = {(r["image"], r["bbox"]): padded_crop(photos[r["image"]], json.loads(r["bbox"])) for r in rows}
    masks = {key: segment(*crop) for key, crop in crops.items()}
    refs = shape_refs(rows, masks)
    for spec in specs:
        qc = get_qc(spec)
        caught = missed = wrong = right = 0
        for r in rows:
            key = (r["image"], r["bbox"])
            v = qc.check(*crops[key], r, refs.get(r["product"]), masks[key])
            rejected, is_bad = not v.ok, key in bad
            caught += rejected and is_bad
            wrong += rejected and not is_bad
            missed += not rejected and is_bad
            right += not rejected and not is_bad
            if rejected != is_bad:
                print(f"  {spec} {'wrongly rejected' if rejected else 'missed'}: {r['image']} p{r['position']} "
                      f"{r['product'][:24]} | check: {v.reason or 'ok'} | by hand: {bad.get(key, 'ok')}")
        print(f"{spec}: caught {caught}/{caught + missed} bad crops, wrongly rejected {wrong}/{wrong + right} good ones\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions")
    ap.add_argument("specs", nargs="+", help="geometry or vlm:<model>")
    args = ap.parse_args()
    main(args.predictions, args.specs)
