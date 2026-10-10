"""Score predictions against the inventory.

Every photo shows all inventory items. Per photo, predicted bottles are matched one-to-one
to inventory items by fuzzy brand+product similarity (Hungarian assignment).

With --alcohol, only alcohol items count (inventory and predictions), as in src/experiment.py.
If data/labelled/fill.csv (image,id,fill_ml: measured, never model-made; blank image = every photo) exists,
also reports fill-level MAE in percentage points of bottle size, over matched bottles whose fill was visible.

Usage: python src/score.py [data/predictions.csv] [--alcohol]
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

BRAND_THRESHOLD = 80  # brand_recall: brands must match, not just product words
MATCH_THRESHOLD = 60  # token_set_ratio below this counts as a wrong identification
BAR = {"spirit", "liqueur", "wine", "beer", "mixer"}  # what a bar camera is for
TO_ML = {"ml": 1, "g": 1, "pint": 568}  # g treated as ml: close enough for sauces
ALCOHOL = {"spirit", "liqueur", "wine", "beer"}
FILL_TRUTH = Path("data/labelled/fill.csv")


def norm(row):
    return f"{row['brand']} {row['product']}".lower()


def brand_ok(pred, item):
    """Strict: the predicted brand names the item's brand. Generic names ("Coffee Liqueur") and "Unknown" fail."""
    p, i = pred["brand"].lower().strip(), item["brand"].lower().strip()
    return bool(p and i and p != "unknown" and fuzz.token_set_ratio(p, i) >= BRAND_THRESHOLD)


def size_ml(row):
    return float(row["size"]) * TO_ML.get(row["unit"], 1)


def match_photo(preds, inventory):
    """For each prediction, the inventory row it is matched to one-to-one, or None."""
    if not preds:
        return []
    sim = np.array([[fuzz.token_set_ratio(norm(p), norm(i)) for i in inventory] for p in preds])
    out = [None] * len(preds)
    for r, c in zip(*linear_sum_assignment(-sim)):
        if sim[r, c] >= MATCH_THRESHOLD:
            out[r] = inventory[c]
    return out


def score_photo(preds, inventory):
    matched = [(p, i) for p, i in zip(preds, match_photo(preds, inventory)) if i is not None]
    n = len(inventory)
    bar = [i for i in inventory if i["category"] in BAR]
    return {
        "count_error": len(preds) - n,
        "recall": len(matched) / n,
        "precision": len(matched) / len(preds) if preds else 0.0,
        "category_acc": sum(p["category"] == i["category"] for p, i in matched) / n,
        "size_acc": sum(abs(size_ml(p) - size_ml(i)) / size_ml(i) <= 0.1 for p, i in matched) / n,
        # strict: matched AND the brand is right; items with no known brand are left out
        "brand_recall": sum(brand_ok(p, i) for p, i in matched) / max(1, sum(bool(i["brand"]) for i in inventory)),
        # recall over the bar items only (spirits, liqueurs, wine, beer, mixers); non-bar predictions ignored
        "bar_recall": sum(i["category"] in BAR for _, i in matched) / len(bar) if bar else 0.0,
    }


def fill_truth(truth, image, item_id):
    return truth.get((image, item_id), truth.get(("", item_id)))


def fill_errors(matched, truth):
    """Absolute fill error in percentage points of bottle size, for matched bottles with a measured fill."""
    return [abs(float(p["fill_ml"]) - fill_truth(truth, p["image"], i["id"])) / size_ml(i) * 100
            for p, i in matched
            if fill_truth(truth, p["image"], i["id"]) is not None and p.get("fill_ml") not in (None, "")]


def main(path, alcohol=False):
    inventory = list(csv.DictReader(open("data/labelled/inventory.csv")))
    if alcohol:
        inventory = [i for i in inventory if i["category"] in ALCOHOL]
    by_photo = defaultdict(list)
    for row in csv.DictReader(open(path)):
        if not alcohol or row["category"] in ALCOHOL:
            by_photo[(row.get("run", "1"), row["image"])].append(row)

    truth = {(r.get("image", ""), r["id"]): float(r["fill_ml"]) for r in csv.DictReader(open(FILL_TRUTH))} if FILL_TRUTH.exists() else {}
    scores, errors, measured = {}, [], 0
    for key, preds in sorted(by_photo.items()):
        scores[key] = score_photo(preds, inventory)
        matched = [(p, i) for p, i in zip(preds, match_photo(preds, inventory)) if i is not None]
        errors += fill_errors(matched, truth)
        measured += sum(fill_truth(truth, p["image"], i["id"]) is not None for p, i in matched)
    metrics = list(next(iter(scores.values())))

    print(f"{'run':>3} {'image':<15}" + "".join(f"{m:>14}" for m in metrics))
    for (run, image), s in scores.items():
        print(f"{run:>3} {image:<15}" + "".join(f"{s[m]:>14.2f}" for m in metrics))

    runs = defaultdict(list)
    for (run, _), s in scores.items():
        runs[run].append(s)
    run_means = {m: [statistics.mean(s[m] for s in photos) for photos in runs.values()] for m in metrics}
    print(f"\n{len(runs)} runs x {len(scores) // len(runs)} photos, {len(inventory)} items per photo")
    for m in metrics:
        vals = run_means[m]
        spread = f" ± {statistics.stdev(vals):.3f} across runs" if len(vals) > 1 else ""
        print(f"  {m:<14}{statistics.mean(vals):>7.3f}{spread}")
    if truth:
        mae = f"{statistics.mean(errors):.1f} pp" if errors else "n/a"
        print(f"  fill MAE      {mae:>7} over {len(errors)}/{measured} matched bottles with a measured fill (rest: fill not visible)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--alcohol"]
    main(args[0] if args else "data/predictions.csv", alcohol="--alcohol" in sys.argv)
