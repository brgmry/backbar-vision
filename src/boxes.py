"""Box and mask helpers. Boxes are [x, y, w, h] (xywh) unless the function name says xyxy."""
import numpy as np
from PIL import Image
from scipy import ndimage

PAD = 0.15  # room around a box when cropping: detector boxes tend to clip caps and bases
DETECT_SIDE = 2400  # detect.py shrinks photos to this long side, so boxes.csv coordinates are at that size


def xyxy(box) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def area_xyxy(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_xyxy(a, b) -> float:
    return area_xyxy([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou_xyxy(a, b) -> float:
    inter = intersection_xyxy(a, b)
    union = area_xyxy(a) + area_xyxy(b) - inter
    return inter / union if union else 0.0


def iou(a, b) -> float:
    return iou_xyxy(xyxy(a), xyxy(b))


def overlap_xyxy(a, b) -> float:
    """Intersection over the smaller box: 1.0 when one box sits inside the other."""
    smaller = min(area_xyxy(a), area_xyxy(b))
    return intersection_xyxy(a, b) / smaller if smaller else 0.0


def padded_crop(photo: Image.Image, box, pad: float = PAD) -> tuple[Image.Image, list[float]]:
    """The photo cropped around `box` with `pad` of its size on each side, and the box in crop coordinates (xyxy)."""
    x, y, w, h = box
    x0, y0 = max(0, round(x - pad * w)), max(0, round(y - pad * h))
    x1, y1 = min(photo.width, round(x + w + pad * w)), min(photo.height, round(y + h + pad * h))
    return photo.crop((x0, y0, x1, y1)), [x - x0, y - y0, x + w - x0, y + h - y0]


def largest_blob(mask: np.ndarray) -> np.ndarray:
    """The largest connected region of a bool mask: one bottle is one blob, the rest is neighbours or reflections."""
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    return labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1


def resized_scale(size: tuple[int, int], side: int) -> float:
    """How much a photo of `size` was shrunk when its long side was capped at `side` (thumbnail-style): multiply
    coordinates in the shrunk copy by this to get original-photo pixels."""
    return max(size) / min(max(size), side)
