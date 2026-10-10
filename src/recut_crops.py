"""Re-cut the labelled crops with whole-bottle boxes (detect.py prefer_whole) and carry the hand labels over.

data/crops/ is left alone. Each new box takes the hand label of the old labelled boxes it overlaps, by geometry
only (no model): old boxes that sit mostly inside the new box, or the old box the new one sits inside. A new box
whose overlapping old boxes disagree on the item, or that overlaps none, gets -1 (excluded from scoring).
These labels are derived, not hand-made: eyeball data/crops_whole/sheets/ before trusting them.

Usage: python src/recut_crops.py   -> data/crops_whole/<photo>/<n>.jpg, boxes.csv, labels.csv, sheets/
Then:  CROPS=data/crops_whole python src/match.py 3
"""
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import Owlv2ForObjectDetection, Owlv2Processor

import detect

OLD, OUT = Path("data/crops"), Path("data/crops_whole")
INSIDE = 0.6  # share of a box's area that must lie inside the other box


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def transfer(new, olds):
    """Item id for a new box from the labelled old boxes of the same photo, plus a short reason."""
    inside = [(inter(new, o), lab) for o, lab in olds if lab >= 0 and inter(new, o) >= INSIDE * area(o)]
    if not inside:
        inside = [(inter(new, o), lab) for o, lab in olds if lab >= 0 and inter(new, o) >= INSIDE * area(new)]
    weight = defaultdict(float)
    for w, lab in inside:
        weight[lab] += w
    if not weight:
        return -1, "no labelled box overlaps"
    items = sorted(weight.items(), key=lambda kv: -kv[1])
    if len(items) > 1 and items[0][0] > 0 and items[1][0] > 0 and items[1][1] > 0.25 * items[0][1]:
        return -1, f"ambiguous: items {items[0][0]} and {items[1][0]}"
    return items[0][0], "transferred"


def main() -> None:
    old_boxes = {(b["photo"], int(b["crop"])): [float(b[k]) for k in ("x1", "y1", "x2", "y2")]
                 for b in csv.DictReader(open(OLD / "boxes.csv"))}
    old = defaultdict(list)
    for r in csv.DictReader(open(OLD / "labels.csv")):
        old[r["photo"]].append((old_boxes[(r["photo"], int(r["crop"]))], int(r["item_id"])))

    processor = Owlv2Processor.from_pretrained(detect.MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(detect.MODEL).to(detect.DEVICE).eval()
    boxes, labels = [], []
    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    for photo in sorted(old):
        img = detect.load(Path("data/raw") / f"{photo}.HEIC")
        dets = sorted(detect.detect(model, processor, img, prefer_whole=True), key=lambda d: d[1][0])
        (OUT / photo).mkdir(exist_ok=True)
        sheet = img.copy()
        draw = ImageDraw.Draw(sheet)
        for n, (score, (x1, y1, x2, y2)) in enumerate(dets, 1):
            box = [max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)]
            img.crop(box).save(OUT / photo / f"{n:02d}.jpg", quality=92)  # same encoding as data/crops
            item, why = transfer(box, old[photo])
            boxes.append({"photo": photo, "crop": n, "score": round(score, 3),
                          **{k: round(v) for k, v in zip(("x1", "y1", "x2", "y2"), box)}})
            labels.append({"photo": photo, "crop": n, "item_id": item, "how": why})
            draw.rectangle(box, outline="red" if item < 0 else "lime", width=4)
            draw.text((box[0] + 4, box[1] + 4), f"{n}:{item}", fill="yellow")
        sheet.save(OUT / "sheets" / f"{photo}.jpg", quality=85)
        print(f"{photo}: {len(dets)} boxes, {sum(l['item_id'] > 0 for l in labels if l['photo'] == photo)} labelled items")
    for name, rows in (("boxes.csv", boxes), ("labels.csv", labels)):
        with (OUT / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
