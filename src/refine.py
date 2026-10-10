"""Grow detector boxes to the whole bottle with SAM 3 outlines.

OWLv2 boxes follow what is easy to see (labels, colour) and stop short of clear glass: a PET base, a clear neck.
One SAM 3 "bottle" call on the band of the photo that holds the boxes gives an outline per bottle; each box grows
to the outline that matches it. Boxes only grow. An outline much wider than its box is a neighbouring bottle (or
two bottles merged) and is ignored.

    boxes = refine(photo, boxes)   # [x, y, w, h] in photo pixels, same order
"""
import numpy as np
from PIL import Image

from fill import sam3_masks

PAD = 0.1  # of the band's height, around the union of the boxes
MIN_IOU = 0.5  # outline vs box: below this the outline belongs to something else
MAX_WIDER = 1.3  # outline width / box width above this = a neighbour got merged in


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def refine(photo: Image.Image, boxes: list[list[int]]) -> list[list[int]]:
    if not boxes:
        return boxes
    xyxy = [[x, y, x + w, y + h] for x, y, w, h in boxes]
    x0, y0 = min(b[0] for b in xyxy), min(b[1] for b in xyxy)
    x1, y1 = max(b[2] for b in xyxy), max(b[3] for b in xyxy)
    pad = PAD * (y1 - y0)
    region = [max(0, round(x0 - pad)), max(0, round(y0 - pad)),
              min(photo.width, round(x1 + pad)), min(photo.height, round(y1 + pad))]
    outlines = []
    for m in sam3_masks(photo.crop(region), "bottle"):
        ys, xs = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
        if ys.size:
            outlines.append([xs[0] + region[0], ys[0] + region[1], xs[-1] + 1 + region[0], ys[-1] + 1 + region[1]])
    out = []
    for b in xyxy:
        best = max(outlines, key=lambda o: iou(b, o), default=None)
        if best is not None and iou(b, best) >= MIN_IOU and (best[2] - best[0]) <= MAX_WIDER * (b[2] - b[0]):
            b = [min(b[0], best[0]), min(b[1], best[1]), max(b[2], best[2]), max(b[3], best[3])]
        out.append([int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])])
    return out
