"""Fill-level estimators. Each gets a full-resolution crop of one identified bottle and returns ml left.

Pick one with a spec string, so experiments swap them from the command line:
  vlm:<model>        the VLM looks at the crop and estimates ml left (the baseline to beat)
  geometry:edge      SAM 2.1 bottle mask, liquid line at the strongest horizontal colour change, volume from the mask
  geometry:labpics   same geometry, liquid segmented by sagieppel's LabPics-trained SAM2 (src/liquid_labpics.py, local)
  geometry:<model>   same geometry, but the VLM points at the liquid line
<model> is any name vlm.get_vlm accepts (opus, sonnet, haiku).

Geometry: the bottle is round, so each pixel row of the mask is a disc of diameter = mask width.
Volume below a row = sum of width^2 down to the base. 100% is the bottom of the neck (first row above
the shoulder, the steepest narrowing above mid-height, where the width levels off): bottles are filled to about there.

Camera tilt: from above, every horizontal circle (liquid surface, base) shows as an ellipse, and its true
height on the bottle axis is the ellipse's middle row, where it meets the side walls. So the liquid level is
taken where the surface meets the walls, and the base is the widest row near the bottom, not the lowest
pixel (that is the front of the base ellipse). The base ellipse's depth also gives the camera elevation.
"""
from dataclasses import dataclass, field
from functools import cache
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel
from scipy import ndimage

from vlm import get_vlm

SAM = "facebook/sam2.1-hiera-large"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SHOULDER_WIN = 0.03  # of bottle height: span over which width change is measured to find the shoulder
NECK_FLAT = 0.1  # neck starts where, above the shoulder, width shrinks by less than 10% per SHOULDER_WIN
MAX_ELEVATION = 60  # degrees; a deeper base ellipse means the base is hidden, not a steeper camera
MIN_CONTRAST = 12.0  # Lab distance; a weaker edge means no visible liquid line
VLM_UPSCALE = 2  # crops are ~400px tall; give the VLM more pixels to look at


@dataclass
class Fill:
    fill_ml: float | None
    visible: bool
    debug: dict = field(default_factory=dict)  # mask, line_y, full_y for overlays


class FillEstimator(Protocol):
    name: str

    def estimate(self, crop: Image.Image, box: list[float], item: dict) -> Fill: ...


def size_ml(item: dict) -> float:
    return float(item["size"]) * {"ml": 1, "g": 1, "pint": 568}.get(item["unit"], 1)


def describe(item: dict) -> str:
    return f"{item['brand']} {item['product']}, {item['size']}{item['unit']}, {item['liquid']}".strip()


# --- VLM estimates ml directly -------------------------------------------------------------------

class Estimate(BaseModel):
    visible: bool
    fill_ml: float | None


class VLMFill:
    def __init__(self, model: str):
        self.vlm = get_vlm(model)
        self.name = f"vlm:{model}"

    def estimate(self, crop, box, item):
        prompt = (f"This photo shows one bottle: {describe(item)}, holding {size_ml(item):g} ml when full. "
                  "visible: can you see the top surface of the liquid through the container? "
                  "fill_ml: ml of liquid left, from the liquid line and the bottle's shape (a neck holds less than "
                  "the body). null when visible is false.")
        ans = self.vlm.ask([upscale(crop)], prompt, Estimate)
        return Fill(ans.fill_ml if ans.visible else None, ans.visible)


# --- geometry: mask + liquid line + volume of revolution ----------------------------------------

class LineFinder(Protocol):
    name: str

    def find(self, crop: Image.Image, mask: np.ndarray, full_y: int, bottom_y: int, item: dict) -> int | None: ...


class Geometry:
    def __init__(self, line: LineFinder):
        self.line = line
        self.name = f"geometry:{line.name}"

    def estimate(self, crop, box, item):
        mask = segment(crop, tuple(box))
        width = row_widths(mask)
        rows = np.flatnonzero(width)
        if rows.size < 20:
            return Fill(None, False, {"mask": mask})
        top, bottom = rows[0], rows[-1]
        full_y = neck_start(width, top, bottom)
        base_y, elevation = base_level(width, top, bottom)
        line_y = self.line.find(crop, mask, full_y, base_y, item)
        debug = {"mask": mask, "full_y": full_y, "base_y": base_y, "line_y": line_y, "elevation": elevation}
        if line_y is None:
            return Fill(None, False, debug)
        line_y = int(np.clip(line_y, full_y, base_y))
        disc = width.astype(float) ** 2
        frac = disc[line_y:base_y + 1].sum() / disc[full_y:base_y + 1].sum()
        return Fill(round(frac * size_ml(item), 1), True, debug)


@cache
def sam():
    from transformers import Sam2Model, Sam2Processor
    return Sam2Processor.from_pretrained(SAM), Sam2Model.from_pretrained(SAM).to(DEVICE).eval()


@cache
def _segment(crop_bytes: bytes, size: tuple[int, int], box: tuple) -> np.ndarray:
    processor, model = sam()
    crop = Image.frombytes("RGB", size, crop_bytes)
    inputs = processor(images=crop, input_boxes=[[list(box)]], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][0].numpy()
    # SAM's top-scoring mask sometimes stops at a dark label or dark liquid; the bottle spans the box's height
    y1, y2 = int(box[1]), int(box[3])
    cover = [m[y1:y2].any(axis=1).mean() for m in masks]
    mask = masks[int(np.argmax(np.array(cover) + 0.01 * out.iou_scores.cpu().numpy().ravel()))]
    labels, n = ndimage.label(mask)
    if n > 1:  # keep the biggest blob: stray bits of neighbouring bottles
        mask = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
    return ndimage.binary_fill_holes(mask)


def segment(crop: Image.Image, box: tuple) -> np.ndarray:
    """Bottle mask inside the crop, prompted with the detector's box. Cached: estimators share it."""
    crop = crop.convert("RGB")
    return _segment(crop.tobytes(), crop.size, box)


def row_widths(mask: np.ndarray) -> np.ndarray:
    """Left-to-right extent per row, so a label printed in the middle doesn't punch holes in the profile."""
    any_px = mask.any(axis=1)
    left = mask.argmax(axis=1)
    right = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    return np.where(any_px, right - left + 1, 0)


def neck_start(width: np.ndarray, top: int, bottom: int) -> int:
    k = max(2, int(SHOULDER_WIN * (bottom - top)))
    w = ndimage.uniform_filter1d(width.astype(float), k)
    ys = np.arange(top + max(k, (bottom - top) // 10), (top + bottom) // 2)  # skip the cap's own top edge
    if ys.size == 0:
        return top
    narrowing = w[ys + k] - w[ys - k]  # how much wider the bottle is just below each row than just above
    y = int(ys[np.argmax(narrowing)])  # steepest point of the shoulder
    while y - k > top and w[y] - w[y - k] > NECK_FLAT * w[y]:
        y -= 1
    return y


def base_level(width: np.ndarray, top: int, bottom: int) -> tuple[int, float]:
    """Row of the base circle's middle (widest row in the bottom quarter) and the camera elevation in degrees."""
    zone = np.arange(bottom - (bottom - top) // 4, bottom + 1)
    widest = width[zone].max()
    base_y = int(zone[width[zone] >= 0.97 * widest].max())
    elevation = float(np.degrees(np.arcsin(np.clip(2 * (bottom - base_y) / widest, 0, 1))))
    if elevation > MAX_ELEVATION:  # base hidden or ragged: fall back to the lowest pixel
        return bottom, None
    return base_y, round(elevation, 1)


class EdgeLine:
    """Strongest change in colour between the band above and the band below, down the middle of the bottle."""
    name = "edge"

    def find(self, crop, mask, full_y, bottom_y, item):
        lab = np.asarray(crop.convert("LAB"), dtype=float)
        h = bottom_y - full_y
        win = max(3, h // 30)
        means = np.full((mask.shape[0], 3), np.nan)
        for y in range(full_y, bottom_y + 1):
            xs = np.flatnonzero(mask[y])
            if xs.size:
                a, b = np.percentile(xs, [30, 70]).astype(int)
                means[y] = lab[y, a:b + 1].mean(axis=0)
        best, best_y = 0.0, None
        for y in range(full_y + win, bottom_y - win - h // 20):  # skip the thick glass base
            d = np.linalg.norm(np.nanmean(means[y - win:y], axis=0) - np.nanmean(means[y:y + win], axis=0))
            if d > best:
                best, best_y = d, y
        return best_y if best >= MIN_CONTRAST else None


class Line(BaseModel):
    visible: bool
    y: int | None


class VLMLine:
    def __init__(self, model: str):
        self.vlm = get_vlm(model)
        self.name = model

    def find(self, crop, mask, full_y, bottom_y, item):
        big = upscale(crop)
        prompt = (f"This {big.width}x{big.height} pixel photo shows one bottle: {describe(item)}. "
                  "visible: can you see the top surface of the liquid through the container? "
                  "y: the pixel row, counted from the top of this image, where the liquid surface meets the left "
                  "and right walls of the bottle. If the camera looks down, the surface shows as an ellipse: give "
                  "the row of its left and right ends (its middle), not its front or back edge. "
                  "null when visible is false.")
        ans = self.vlm.ask([big], prompt, Line)
        return ans.y // VLM_UPSCALE if ans.visible and ans.y is not None else None


def upscale(crop: Image.Image) -> Image.Image:
    return crop.resize((crop.width * VLM_UPSCALE, crop.height * VLM_UPSCALE), Image.LANCZOS)


def get_estimator(spec: str) -> FillEstimator:
    kind, _, arg = spec.partition(":")
    if kind == "vlm":
        return VLMFill(arg)
    if kind == "geometry" and arg == "labpics":
        from liquid_labpics import LabPicsLine  # needs the vendored model, see that file
        return Geometry(LabPicsLine())
    if kind == "geometry":
        return Geometry(EdgeLine() if arg == "edge" else VLMLine(arg))
    raise ValueError(f"unknown fill estimator {spec!r}: use vlm:<model>, geometry:edge, geometry:labpics or geometry:<model>")
