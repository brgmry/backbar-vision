"""Crop clearing: reject crops that can't support a fill reading (cut off, hidden, wrong thing) before fill runs.

Pick one with a spec string, like the fill estimators:
  geometry       local and free, from the SAM mask: touches the crop's top or bottom (cut off), no neck,
                 mask fills too little of its box, or height/width far from the same item in other photos
  vlm:<model>    ask the VLM whether the whole bottle is visible, cap to base, nothing in front

Scored against the hand exclusions (data/labelled/crop_exclude.csv):
  python src/qc.py <predictions.csv> <qc spec> [<qc spec> ...]
"""
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from pydantic import BaseModel

from fill import describe, fit, neck_start, row_widths, segment
from vlm import get_vlm

register_heif_opener()

EXCLUDE = Path("data/labelled/crop_exclude.csv")  # hand-rejected crops: ground truth for scoring the check only

EDGE_PX = 3  # mask within this many px of the crop's top or bottom = bottle runs out of the crop
MIN_FILL = 0.55  # mask area / box area; below this the bottle is hidden or the box is wrong
MAX_SHAPE_DEV = 0.25  # height/width ratio vs the item's median across photos


class Verdict(BaseModel):
    ok: bool
    reason: str


class CropQC(Protocol):
    name: str

    def check(self, crop: Image.Image, box: list[float], item: dict, shape_ref: float | None) -> Verdict: ...


def shape(crop: Image.Image, box: list[float]) -> tuple[np.ndarray, float | None]:
    mask = segment(crop, tuple(box))
    width = row_widths(mask)
    rows = np.flatnonzero(width)
    if rows.size < 20:
        return mask, None
    body = np.median(width[(rows[0] + rows[-1]) // 2:rows[-1] + 1])
    return mask, (rows[-1] - rows[0]) / body


class GeometryQC:
    name = "geometry"

    def check(self, crop, box, item, shape_ref):
        mask, ratio = shape(crop, box)
        if ratio is None:
            return Verdict(ok=False, reason="no bottle mask")
        rows = np.flatnonzero(mask.any(axis=1))
        if rows[0] < EDGE_PX or rows[-1] > mask.shape[0] - 1 - EDGE_PX:
            return Verdict(ok=False, reason="bottle runs out of the crop")
        width = row_widths(mask)
        if neck_start(width, rows[0], rows[-1]) <= rows[0]:
            return Verdict(ok=False, reason="no neck found")
        x1, y1, x2, y2 = box
        if mask.sum() / ((x2 - x1) * (y2 - y1)) < MIN_FILL:
            return Verdict(ok=False, reason="mask fills too little of the box")
        if shape_ref and abs(ratio / shape_ref - 1) > MAX_SHAPE_DEV:
            return Verdict(ok=False, reason=f"shape {ratio:.2f} vs usual {shape_ref:.2f}")
        return Verdict(ok=True, reason="")


class VLMQC:
    def __init__(self, model: str):
        self.vlm = get_vlm(model)
        self.name = f"vlm:{model}"

    def check(self, crop, box, item, shape_ref):
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


def shape_refs(rows: list[dict], crops: dict) -> dict[str, float]:
    """Median height/width per item across all its crops: a cut-off or hidden bottle stands out against it."""
    by_item = {}
    for r in rows:
        _, ratio = shape(*crops[(r["image"], r["bbox"])])
        if ratio:
            by_item.setdefault(r["product"], []).append(ratio)
    return {k: statistics.median(v) for k, v in by_item.items()}


def main(ids_path: str, specs: list[str]) -> None:
    from experiment import ALCOHOL, padded_crop
    rows = [r for r in csv.DictReader(open(ids_path)) if r["category"] in ALCOHOL]
    bad = {(r["image"], r["bbox"]): r["reason"] for r in csv.DictReader(open(EXCLUDE))}
    photos = {img: Image.open(Path("data/raw") / img).convert("RGB") for img in {r["image"] for r in rows}}
    crops = {(r["image"], r["bbox"]): padded_crop(photos[r["image"]], json.loads(r["bbox"])) for r in rows}
    refs = shape_refs(rows, crops)
    for spec in specs:
        qc = get_qc(spec)
        tp = fp = fn = tn = 0
        for r in rows:
            key = (r["image"], r["bbox"])
            v = qc.check(*crops[key], r, refs.get(r["product"]))
            rejected, is_bad = not v.ok, key in bad
            tp, fp, fn, tn = tp + (rejected and is_bad), fp + (rejected and not is_bad), fn + (not rejected and is_bad), tn + (not rejected and not is_bad)
            if rejected != is_bad:
                print(f"  {spec} {'wrongly rejected' if rejected else 'missed'}: {r['image']} p{r['position']} "
                      f"{r['product'][:24]} | qc: {v.reason or 'ok'} | hand: {bad.get(key, 'ok')}")
        print(f"{spec}: caught {tp}/{tp + fn} bad crops, wrongly rejected {fp}/{fp + tn} good ones\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
