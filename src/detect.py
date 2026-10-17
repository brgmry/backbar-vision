"""Find bottles with OWLv2 and save the crops, for the catalogue experiments (the API uses find.py).

Usage: python src/detect.py [photos_dir] [out_dir]   (defaults: data/raw, data/crops)
  -> <out_dir>/<photo>/<n>.jpg and <out_dir>/boxes.csv, boxes in the photo shrunk to boxes.DETECT_SIDE
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from boxes import DETECT_SIDE, iou_xyxy, overlap_xyxy
from models import DEVICE
from photos import EXTS, open_photo

MODEL = "google/owlv2-base-patch16-ensemble"
QUERIES = ["a bottle", "a jug", "a sauce bottle"]
SCORE_THRESHOLD = 0.12
TILES = 3  # OWLv2 sees 960px; tile the photo so small bottles keep enough pixels
OVERLAP = 0.4
EDGE = 4  # px
NMS_IOU = 0.3
NMS_CONTAINED = 0.7  # a box this much inside a kept box is a part of that bottle (cap, label)


def load(path: Path) -> Image.Image:
    img = open_photo(path)
    img.thumbnail((DETECT_SIDE, DETECT_SIDE))
    return img


def load_model() -> tuple[Owlv2ForObjectDetection, Owlv2Processor]:
    return Owlv2ForObjectDetection.from_pretrained(MODEL).to(DEVICE).eval(), Owlv2Processor.from_pretrained(MODEL)


def detect(model, processor, img: Image.Image, prefer_whole: bool = False) -> list[tuple[float, list[float]]]:
    """prefer_whole: boxes from the whole-image pass win over tile boxes, whatever their score. Tiles often see
    a bottle in pieces (cap half, label half) that outscore the whole-bottle box; with this on, tile boxes only
    add bottles the whole-image pass missed (small ones)."""
    w, h = img.size
    tw, th = w / (TILES - (TILES - 1) * OVERLAP), h / (TILES - (TILES - 1) * OVERLAP)
    boxes, scores = [], []
    for i in range(TILES):
        for j in range(TILES):
            x0, y0 = i * tw * (1 - OVERLAP), j * th * (1 - OVERLAP)
            for score, (x1, y1, x2, y2) in detect_tile(model, processor, img.crop((x0, y0, x0 + tw, y0 + th))):
                # a box touching an inner tile edge is a bottle cut in half; another tile has it whole
                if (x1 < EDGE and i > 0) or (y1 < EDGE and j > 0) or (x2 > tw - EDGE and i < TILES - 1) or (y2 > th - EDGE and j < TILES - 1):
                    continue
                boxes.append([x1 + x0, y1 + y0, x2 + x0, y2 + y0])
                scores.append(score)
    n_tiles = len(boxes)
    for score, box in detect_tile(model, processor, img):
        boxes.append(box)
        scores.append(score)
    if not boxes:
        return []
    rank = [s + (1.0 if prefer_whole and i >= n_tiles else 0.0) for i, s in enumerate(scores)]  # scores are < 1
    return [(scores[i], boxes[i]) for i in nms(boxes, rank)]


def detect_tile(model, processor, img: Image.Image) -> list[tuple[float, list[float]]]:
    inputs = processor(text=[QUERIES], images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    # OWLv2 pads to a square, so boxes are relative to the longer side
    side = max(img.size)
    result = processor.post_process_grounded_object_detection(
        outputs, threshold=SCORE_THRESHOLD, target_sizes=torch.tensor([[side, side]])
    )[0]
    return [(float(s), b.tolist()) for s, b in zip(result["scores"].cpu(), result["boxes"].cpu(), strict=True)]


def nms(boxes: list[list[float]], rank: list[float]) -> list[int]:
    """Indices of the boxes kept, best rank first: a box goes if it overlaps or sits inside one already kept."""
    keep = []
    for i in sorted(range(len(boxes)), key=lambda i: -rank[i]):
        if all(iou_xyxy(boxes[i], boxes[j]) < NMS_IOU and overlap_xyxy(boxes[i], boxes[j]) < NMS_CONTAINED
               for j in keep):
            keep.append(i)
    return keep


def main(folder: str, out_dir: str) -> None:
    model, processor = load_model()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS):
        img = load(path)
        dets = sorted(detect(model, processor, img), key=lambda d: d[1][0])  # left to right
        (out / path.stem).mkdir(exist_ok=True)
        for n, (score, (x1, y1, x2, y2)) in enumerate(dets, 1):
            box = [max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)]
            img.crop(box).save(out / path.stem / f"{n:02d}.jpg", quality=92)
            rows.append({"photo": path.stem, "crop": n, "score": round(score, 3),
                         "x1": round(box[0]), "y1": round(box[1]), "x2": round(box[2]), "y2": round(box[3])})
        print(f"{path.name}: {len(dets)} bottles")
    with (out / "boxes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("photos", nargs="?", default="data/raw")
    ap.add_argument("out", nargs="?", default="data/crops")
    args = ap.parse_args()
    main(args.photos, args.out)
