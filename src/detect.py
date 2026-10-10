"""Step 1: find every bottle in each photo with OWLv2 and save the crops.

Usage: python src/detect.py [photos_dir] [out_dir]   -> <out_dir>/<photo>/<n>.jpg + <out_dir>/boxes.csv (default data/raw -> data/crops)
"""
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from pillow_heif import register_heif_opener
from transformers import Owlv2ForObjectDetection, Owlv2Processor

register_heif_opener()

MODEL = "google/owlv2-base-patch16-ensemble"
QUERIES = ["a bottle", "a jug", "a sauce bottle"]
SCORE_THRESHOLD = 0.12
TILES = 3  # OWLv2 sees 960px; tile the photo so small bottles keep enough pixels
OVERLAP = 0.4
EDGE = 4  # px
NMS_IOU = 0.3
MAX_SIDE = 2400
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def detect(model, processor, img: Image.Image) -> list[tuple[float, list[float]]]:
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
    for score, box in detect_tile(model, processor, img):
        boxes.append(box)
        scores.append(score)
    if not boxes:
        return []
    boxes_t, scores_t = torch.tensor(boxes), torch.tensor(scores)
    return [(scores[i], boxes[i]) for i in nms(boxes_t, scores_t)]


def detect_tile(model, processor, img: Image.Image) -> list[tuple[float, list[float]]]:
    inputs = processor(text=[QUERIES], images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    # OWLv2 pads to a square, so boxes are relative to the longer side
    side = max(img.size)
    result = processor.post_process_grounded_object_detection(
        outputs, threshold=SCORE_THRESHOLD, target_sizes=torch.tensor([[side, side]])
    )[0]
    return [(float(s), b.tolist()) for s, b in zip(result["scores"].cpu(), result["boxes"].cpu())]


def nms(boxes: torch.Tensor, scores: torch.Tensor) -> list[int]:
    order, keep = scores.argsort(descending=True).tolist(), []
    for i in order:
        if all(iou(boxes[i], boxes[j]) < NMS_IOU and overlap(boxes[i], boxes[j]) < 0.7 for j in keep):
            keep.append(i)
    return keep


def overlap(a, b) -> float:
    """Intersection over the smaller box."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / smaller) if smaller else 0.0


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union else 0.0


def main(folder: str, out_dir: str = "data/crops") -> None:
    processor = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(DEVICE).eval()
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
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw", sys.argv[2] if len(sys.argv) > 2 else "data/crops")
