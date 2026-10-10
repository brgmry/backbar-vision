"""Find every bottle in a photo with SAM 3 (text prompt "bottle" on the whole photo): one outline per bottle.

No identification here; that is a separate step. Bottles come back in reading order (shelf rows top to bottom,
then left to right), each with its box in photo pixels and its outline as a mask over that box.

    for bottle in find_bottles(photo): bottle.box, bottle.outline(crop_size, box_in_crop), bottle.upside_down
"""
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from fill import DEVICE, sam3

SIDE = 1500  # the photo is shrunk to this long side first (SAM 3 works at 1008 px internally anyway)
MIN_HEIGHT = 0.03  # of the photo's height: smaller masks are caps, labels, reflections
ROW_GAP = 0.5  # of the median bottle height: a centre this far below the row's first bottle starts a new row
INVERTED = 1.6  # top of the outline this much wider than its bottom = an upside-down (optic) bottle
MAX_WIDE = 1.2  # an outline this much wider than tall is a group of bottles (or a lying one), not a bottle


@dataclass
class Found:
    box: list[int]  # [x, y, w, h] in photo pixels
    mask: np.ndarray  # outline over the box, at the shrunk scale
    score: float

    @property
    def upside_down(self) -> bool:
        rows = np.flatnonzero(self.mask.any(axis=1))
        if rows.size < 10:
            return False
        width = self.mask.sum(axis=1)[rows[0]:rows[-1] + 1]
        k = max(1, len(width) // 7)
        return float(np.median(width[:k])) > INVERTED * float(np.median(width[-k:]))

    def outline(self, crop_size: tuple[int, int], box_in_crop: list[float]) -> np.ndarray:
        """The outline as a bool mask over a crop of the photo, given where the box sits in that crop."""
        x, y = round(box_in_crop[0]), round(box_in_crop[1])
        w, h = self.box[2], self.box[3]
        up = np.asarray(Image.fromarray(self.mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
        out = np.zeros((crop_size[1], crop_size[0]), bool)
        out[y:y + h, x:x + w] = up[:max(0, crop_size[1] - y), :max(0, crop_size[0] - x)]
        return out


def find_bottles(photo: Image.Image) -> list[Found]:
    processor, model = sam3()
    scale = min(1.0, SIDE / max(photo.size))
    small = photo.resize((round(photo.width * scale), round(photo.height * scale))) if scale < 1 else photo
    inputs = processor(images=small, text="bottle", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    res = processor.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5,
                                                       target_sizes=[[small.height, small.width]])[0]
    found = []
    for m, s in zip(res["masks"].cpu().numpy().astype(bool), res["scores"].cpu().tolist()):
        labels, n = ndimage.label(m)
        if n > 1:  # one bottle is one blob; stray pieces elsewhere (another shelf, a reflection) go
            m = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
        ys, xs = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        if ys.size == 0 or (ys[-1] - ys[0] + 1) < MIN_HEIGHT * small.height:
            continue
        if (xs[-1] - xs[0] + 1) > MAX_WIDE * (ys[-1] - ys[0] + 1):
            continue
        x0, x1, y0, y1 = xs[0], xs[-1] + 1, ys[0], ys[-1] + 1
        box = [round(x0 / scale), round(y0 / scale), round((x1 - x0) / scale), round((y1 - y0) / scale)]
        found.append(Found(box, m[y0:y1, x0:x1], s))
    return reading_order(found)


def reading_order(found: list[Found]) -> list[Found]:
    if not found:
        return found
    gap = ROW_GAP * float(np.median([f.box[3] for f in found]))
    rows: list[list[Found]] = []
    for f in sorted(found, key=lambda f: f.box[1] + f.box[3] / 2):
        centre = f.box[1] + f.box[3] / 2
        if rows and centre - (rows[-1][0].box[1] + rows[-1][0].box[3] / 2) <= gap:
            rows[-1].append(f)
        else:
            rows.append([f])
    return [f for row in rows for f in sorted(row, key=lambda f: f.box[0])]
