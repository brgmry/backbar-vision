"""Fill-level estimators. Each takes a full-resolution crop of one bottle and returns the ml left.

Pick one with a spec string, so experiments swap them from the command line:
  vlm:<model>            the VLM looks at the crop and estimates the ml left (the baseline to beat)
  geometry:edge          liquid line at the strongest horizontal colour change
  geometry:labpics       liquid segmented by sagieppel's LabPics-trained SAM 2 (src/liquid_labpics.py)
  geometry:sam3          liquid segmented by SAM 3 from the text prompt "liquid"
  geometry:<model>       the VLM gives the pixel row of the liquid line
  geometry:<model>-grid  numbered ticks are drawn beside the bottle and the VLM names the tick at the liquid line
<model> is a name vlm.get_vlm accepts (opus, sonnet, haiku, or a full Claude model id).

Geometry estimators share one method. The bottle outline comes from SAM 3 ("bottle" prompt, the instance that best
matches the detector's box), with SAM 2.1 prompted by the box as fallback; OUTLINE=sam2 uses SAM 2.1 only. Callers
that already have the outline (the API, from its whole-photo SAM 3 pass) pass it in as `mask`.
The bottle is round, so each pixel row of the outline is a disc as wide as the outline: the volume below a row is
the sum of width^2 down to the base. 100% is the bottom of the neck, found as the steepest narrowing above
mid-height followed up to where the width levels off; bottles are filled to about there.

Camera tilt: seen from above, every horizontal circle (the liquid surface, the base) is an ellipse, and its true
height on the bottle's axis is the ellipse's middle row, where it meets the side walls. So the liquid level is read
where the surface meets the walls, and the base is the widest row near the bottom rather than the lowest pixel
(that is the front of the base ellipse). The depth of the base ellipse also gives the camera's elevation.
"""
import os
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
from scipy import ndimage

from boxes import largest_blob
from models import DEVICE, resize_mask, sam2, sam3_instances
from schema import size_ml
from vlm import MAX_PIXELS, MAX_SIDE, get_vlm, known_model

OUTLINE = os.environ.get("OUTLINE", "sam3")
SAM3_SIDE = 1200  # long side of a crop fed to SAM 3; larger is slower and no sharper
MIN_BOX_IOU = 0.3  # a SAM 3 instance overlapping the detector's box less than this is a neighbour
COVER_TIEBREAK = 0.01  # SAM 2.1 masks are picked by how much of the box height they cover; its own score breaks ties
MIN_ROWS = 20  # an outline shorter than this many pixel rows is too small to measure
SHOULDER_WIN = 0.03  # of bottle height: span over which width change is measured to find the shoulder
NECK_FLAT = 0.1  # the neck starts where, above the shoulder, width shrinks by less than 10% per SHOULDER_WIN
CAP_SKIP = 0.1  # of bottle height: the shoulder search starts below the cap's own top edge
BASE_ZONE = 0.25  # of bottle height: the base is looked for in the bottom quarter
BASE_WIDEST = 0.97  # rows at least this share of the widest bottom row count as the base circle
MAX_ELEVATION = 60  # degrees; a deeper base ellipse means the base is hidden, not a steeper camera
BAND = 1 / 30  # of the liquid column height: rows averaged above and below a candidate line
GLASS_BASE = 0.05  # of the liquid column height: the thick glass base, skipped by the edge finder
MIDDLE = (30, 70)  # percentiles of each row's width averaged for its colour: the middle, away from the glass walls
MIN_EDGE_CONTRAST = 12.0  # Lab distance; the edge finder reports no line below this
MIN_LINE_CONTRAST = float(os.environ.get("MIN_LINE_CONTRAST", 15))  # a found line weaker than this gives "unsure"
# 15 sits above the two weak, clearly wrong readings on our photos (contrast 11-12) and below every reading
# measured right (21 and up)
MIN_LIQUID_SPECK = 0.05  # SAM 3 "liquid" pieces smaller than this share of all liquid are reflections
GRID_TICKS = 30  # numbered ticks from the neck (1) to the base (GRID_TICKS)
VLM_UPSCALE = 2  # small crops (~400 px tall) are enlarged up to 2x for the VLM; big ones shrink to its limit


class UnknownEstimator(ValueError):
    """A fill spec that names no estimator, or a model vlm.get_vlm does not know."""


@dataclass
class Fill:
    fill_ml: float | None
    visible: bool
    debug: dict = field(default_factory=dict)  # outline, line rows and camera angle, for overlays
    note: str | None = None  # why fill_ml is null, when it is


class FillEstimator(Protocol):
    name: str
    uses_torch: bool  # runs a local model in estimate(), so callers serialise it with other GPU work
    uses_api: bool  # calls a hosted model, so it needs an API key

    def estimate(self, crop: Image.Image, box: list[float], item: dict, mask: np.ndarray | None = None) -> Fill: ...


def describe(item: dict) -> str:
    if not item.get("product"):
        return "an unidentified bottle"
    return f"{item['brand'] or ''} {item['product']}, {item['size']}{item['unit']}, {item['liquid']}".strip()


# VLM estimates the ml directly

class Estimate(BaseModel):
    visible: bool
    fill_ml: float | None


class VLMFill:
    uses_torch = False
    uses_api = True

    def __init__(self, model: str):
        self.vlm = get_vlm(model)
        self.name = f"vlm:{model}"

    def estimate(self, crop, box, item, mask=None):
        prompt = (f"This photo shows one bottle: {describe(item)}, holding {size_ml(item):g} ml when full. "
                  "visible: can you see the top surface of the liquid through the container? "
                  "fill_ml: ml of liquid left, from the liquid line and the bottle's shape (a neck holds less than "
                  "the body). null when visible is false.")
        ans = self.vlm.ask([fit(crop)[0]], prompt, Estimate)
        return Fill(ans.fill_ml if ans.visible else None, ans.visible)


# Geometry: outline, liquid line, volume of revolution

class LineFinder(Protocol):
    name: str
    uses_torch: bool
    uses_api: bool

    def find(self, crop: Image.Image, mask: np.ndarray, full_y: int, base_y: int, item: dict) -> int | None: ...


class Geometry:
    def __init__(self, line: LineFinder):
        self.line = line
        self.name = f"geometry:{line.name}"
        self.uses_torch = line.uses_torch
        self.uses_api = line.uses_api

    def estimate(self, crop, box, item, mask=None):
        mask = segment(crop, box) if mask is None else mask
        width = row_widths(mask)
        rows = np.flatnonzero(width)
        if rows.size < MIN_ROWS:
            return Fill(None, False, {"mask": mask}, "bottle outline too small to measure")
        top, bottom = rows[0], rows[-1]
        full_y = neck_start(width, top, bottom)
        base_y, elevation = base_level(width, top, bottom)
        line_y = self.line.find(crop, mask, full_y, base_y, item)
        debug = {"mask": mask, "full_y": full_y, "base_y": base_y, "line_y": line_y, "elevation": elevation,
                 **getattr(self.line, "debug", {})}
        if line_y is None:
            return Fill(None, False, debug, "liquid line not visible")
        line_y = int(np.clip(line_y, full_y, base_y))
        debug["contrast"] = contrast = line_contrast(crop, mask, line_y, full_y, base_y)
        if contrast < MIN_LINE_CONTRAST:
            return Fill(None, False, debug, f"unsure: no sharp liquid line ({contrast:.0f} < {MIN_LINE_CONTRAST:g})")
        disc = width.astype(float) ** 2
        frac = disc[line_y:base_y + 1].sum() / disc[full_y:base_y + 1].sum()
        return Fill(round(frac * size_ml(item), 1), True, debug)


def segment(crop: Image.Image, box) -> np.ndarray:
    """Bottle outline inside the crop, `box` being the detector's box in crop coordinates (xyxy)."""
    crop = crop.convert("RGB")
    mask = outline_sam3(crop, box) if OUTLINE == "sam3" else None
    return mask if mask is not None else outline_sam2(crop, box)


def outline_sam2(crop: Image.Image, box) -> np.ndarray:
    processor, model = sam2()
    inputs = processor(images=crop, input_boxes=[[list(box)]], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][0].numpy()
    # SAM's top-scoring mask sometimes stops at a dark label or dark liquid, while the bottle spans the box's height
    y1, y2 = int(box[1]), int(box[3])
    cover = np.array([m[y1:y2].any(axis=1).mean() for m in masks])
    mask = masks[int(np.argmax(cover + COVER_TIEBREAK * out.iou_scores.cpu().numpy().ravel()))]
    return ndimage.binary_fill_holes(largest_blob(mask))


def outline_sam3(crop: Image.Image, box) -> np.ndarray | None:
    """The SAM 3 "bottle" instance that best overlaps the detector's box, or None if none overlaps enough."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    inbox = np.zeros((crop.height, crop.width), bool)
    inbox[max(0, y1):y2, max(0, x1):x2] = True
    masks, _, _ = sam3_instances(crop, "bottle", SAM3_SIDE)
    best, best_iou = None, MIN_BOX_IOU
    for small in masks:
        m = resize_mask(small, crop.size)
        iou = (m & inbox).sum() / (m | inbox).sum()
        if iou > best_iou:
            best, best_iou = m, iou
    return None if best is None else ndimage.binary_fill_holes(largest_blob(best))


def row_widths(mask: np.ndarray) -> np.ndarray:
    """Left-to-right extent per row, so a label printed in the middle doesn't punch holes in the profile."""
    any_px = mask.any(axis=1)
    left = mask.argmax(axis=1)
    right = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    return np.where(any_px, right - left + 1, 0)


def neck_start(width: np.ndarray, top: int, bottom: int) -> int:
    k = max(2, int(SHOULDER_WIN * (bottom - top)))
    w = ndimage.uniform_filter1d(width.astype(float), k)
    ys = np.arange(top + max(k, int(CAP_SKIP * (bottom - top))), (top + bottom) // 2)
    if ys.size == 0:
        return top
    narrowing = w[ys + k] - w[ys - k]  # how much wider the bottle is just below each row than just above
    y = int(ys[np.argmax(narrowing)])  # steepest point of the shoulder
    while y - k > top and w[y] - w[y - k] > NECK_FLAT * w[y]:
        y -= 1
    return y


def base_level(width: np.ndarray, top: int, bottom: int) -> tuple[int, float | None]:
    """Row of the base circle's middle (the widest rows near the bottom) and the camera elevation in degrees.
    A base ellipse too deep to be a camera angle means the base is hidden: the lowest row and no angle."""
    zone = np.arange(bottom - int(BASE_ZONE * (bottom - top)), bottom + 1)
    widest = width[zone].max()
    base_y = int(zone[width[zone] >= BASE_WIDEST * widest].max())
    elevation = float(np.degrees(np.arcsin(np.clip(2 * (bottom - base_y) / widest, 0, 1))))
    if elevation > MAX_ELEVATION:
        return bottom, None
    return base_y, round(elevation, 1)


def row_colours(crop: Image.Image, mask: np.ndarray, rows) -> dict[int, np.ndarray]:
    """Mean Lab colour of the middle of each row of the outline, away from reflections on the glass walls."""
    lab = np.asarray(crop.convert("LAB"), dtype=float)
    out = {}
    for r in rows:
        xs = np.flatnonzero(mask[r]) if 0 <= r < mask.shape[0] else []
        if len(xs):
            a, b = np.percentile(xs, MIDDLE).astype(int)
            out[r] = lab[r, a:b + 1].mean(axis=0)
    return out


def band_change(colours: dict[int, np.ndarray], r: int, w: int) -> float | None:
    """Lab distance between the mean colour of the `w` rows above row r and the `w` rows from r down."""
    above = [colours[i] for i in range(r - w, r) if i in colours]
    below = [colours[i] for i in range(r, r + w) if i in colours]
    if not above or not below:
        return None
    return float(np.linalg.norm(np.mean(above, axis=0) - np.mean(below, axis=0)))


def line_contrast(crop: Image.Image, mask: np.ndarray, y: int, full_y: int, base_y: int) -> float:
    """How sharp the liquid line is: the largest colour change within a few rows of y (a found line can be a little
    off). A line at the neck of a full bottle is measured the same way, against the air in the neck."""
    w = max(3, int(BAND * (base_y - full_y)))
    colours = row_colours(crop, mask, range(y - 2 * w, y + 2 * w + 1))
    changes = [band_change(colours, r, w) for r in range(y - w, y + w + 1)]
    return max((c for c in changes if c is not None), default=0.0)


class EdgeLine:
    """The strongest colour change between the band above and the band below, down the middle of the bottle."""
    name = "edge"
    uses_torch = False
    uses_api = False

    def find(self, crop, mask, full_y, base_y, item):
        h = base_y - full_y
        w = max(3, int(BAND * h))
        colours = row_colours(crop, mask, range(full_y, base_y + 1))
        best, best_y = 0.0, None
        for y in range(full_y + w, base_y - w - int(GLASS_BASE * h)):
            change = band_change(colours, y, w)
            if change is not None and change > best:
                best, best_y = change, y
        return best_y if best >= MIN_EDGE_CONTRAST else None


class Line(BaseModel):
    visible: bool
    y: int | None


class VLMLine:
    uses_torch = False
    uses_api = True

    def __init__(self, model: str):
        self.vlm = get_vlm(model)
        self.name = model

    def find(self, crop, mask, full_y, base_y, item):
        big, scale = fit(crop)
        prompt = (f"This {big.width}x{big.height} pixel photo shows one bottle: {describe(item)}. "
                  "visible: can you see the top surface of the liquid through the container? "
                  "y: the pixel row, counted from the top of this image, where the liquid surface meets the left "
                  "and right walls of the bottle. If the camera looks down, the surface shows as an ellipse: give "
                  "the row of its left and right ends (its middle), not its front or back edge. "
                  "null when visible is false.")
        ans = self.vlm.ask([big], prompt, Line)
        return round(ans.y / scale) if ans.visible and ans.y is not None else None


class Tick(BaseModel):
    visible: bool
    tick: int | None


class VLMGridLine:
    """Numbered ticks beside the bottle, neck to base; the VLM names the tick level with the liquid surface.
    Naming a printed number is easier for a VLM than giving a pixel coordinate."""
    uses_torch = False
    uses_api = True

    def __init__(self, model: str):
        self.vlm = get_vlm(model)
        self.name = f"{model}-grid"

    def find(self, crop, mask, full_y, base_y, item):
        ys = np.linspace(full_y, base_y, GRID_TICKS).round().astype(int)
        big, scale = fit(crop)
        cols = np.flatnonzero(mask.any(axis=0))
        left, right = cols[0] * scale, cols[-1] * scale
        draw = ImageDraw.Draw(big)
        font = ImageFont.load_default(size=max(10, int(big.height / 70)))
        tick = max(8, int(big.width / 25))
        for n, y in enumerate(ys, 1):
            y = y * scale
            colour = "red" if n % 2 else "yellow"
            draw.line([(max(0, left - tick), y), (left - 2, y)], fill=colour, width=2)
            draw.line([(right + 2, y), (min(big.width, right + tick), y)], fill=colour, width=2)
            if n % 2:
                draw.text((max(0, left - tick - font.size * 1.4), y - font.size / 2), str(n), fill=colour, font=font)
        prompt = (f"This photo shows one bottle: {describe(item)}. Numbered ticks run down both sides of the "
                  f"bottle, 1 at the neck (full) to {GRID_TICKS} at the base (empty); odd numbers are printed, even "
                  "ticks sit between them. visible: can you see the top surface of the liquid through the container? "
                  "tick: the tick level with where the liquid surface meets the side walls. If the camera looks down "
                  "the surface is an ellipse: use its left and right ends, not its front or back edge. "
                  "1 if the liquid reaches the neck. null when visible is false.")
        ans = self.vlm.ask([big], prompt, Tick)
        if not ans.visible or ans.tick is None:
            return None
        return int(ys[int(np.clip(ans.tick, 1, GRID_TICKS)) - 1])


class SAM3Line:
    """Liquid = every SAM 3 "liquid" instance inside the bottle outline; the level is where it meets the walls."""
    name = "sam3"
    uses_torch = True
    uses_api = False

    def __init__(self):
        self.debug: dict = {}

    def find(self, crop, mask, full_y, base_y, item):
        from liquid_labpics import MIN_LIQUID, wall_level
        liquid = np.zeros_like(mask)
        for small in sam3_instances(crop, "liquid", SAM3_SIDE)[0]:
            liquid |= resize_mask(small, crop.size) & mask
        labels, n = ndimage.label(liquid)
        if n > 1:  # drop specks (reflections) but keep pieces a label splits apart
            sizes = np.bincount(labels.ravel())[1:]
            liquid = np.isin(labels, np.flatnonzero(sizes >= MIN_LIQUID_SPECK * sizes.sum()) + 1)
        self.debug = {"liquid": liquid}
        if liquid.sum() < MIN_LIQUID * mask.sum():
            return None
        return wall_level(liquid, mask, full_y)


def fit(crop: Image.Image) -> tuple[Image.Image, float]:
    """The crop resized for the VLM (up to VLM_UPSCALE, within its size limit), and the factor used."""
    scale = min(VLM_UPSCALE, MAX_SIDE / max(crop.size), (MAX_PIXELS / (crop.width * crop.height)) ** 0.5)
    return crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS), scale


def get_estimator(spec: str) -> FillEstimator:
    kind, _, arg = spec.partition(":")
    if kind == "geometry" and arg in ("edge", "sam3", "labpics"):
        if arg == "labpics":
            from liquid_labpics import LabPicsLine  # needs the vendored model, see that file
            return Geometry(LabPicsLine())
        return Geometry(EdgeLine() if arg == "edge" else SAM3Line())
    model = arg.removesuffix("-grid")
    if kind in ("vlm", "geometry") and known_model(model):
        if kind == "vlm":
            return VLMFill(model)
        return Geometry(VLMGridLine(model) if arg.endswith("-grid") else VLMLine(model))
    raise UnknownEstimator(f"unknown fill estimator {spec!r}: use vlm:<model>, geometry:edge, geometry:labpics, "
                           "geometry:sam3, geometry:<model> or geometry:<model>-grid, <model> one of opus, sonnet, "
                           "haiku or a claude-* model id")
