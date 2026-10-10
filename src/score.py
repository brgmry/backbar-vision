"""Score predictions against the inventory.

Every photo shows all inventory items. Per photo, predicted bottles are matched one-to-one
to inventory items by fuzzy brand+product similarity (Hungarian assignment).

Usage: python src/score.py [data/predictions.csv]
"""
import csv
import statistics
import sys
from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

MATCH_THRESHOLD = 60  # token_set_ratio below this counts as a wrong identification
BAR = {"spirit", "liqueur", "wine", "beer", "mixer"}  # what a bar camera is for
TO_ML = {"ml": 1, "g": 1, "pint": 568}  # g treated as ml: close enough for sauces


def norm(row):
    return f"{row['brand']} {row['product']}".lower()


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
        # recall over the bar items only (spirits, liqueurs, wine, beer, mixers); non-bar predictions ignored
        "bar_recall": sum(i["category"] in BAR for _, i in matched) / len(bar) if bar else 0.0,
    }


def main(path):
    inventory = list(csv.DictReader(open("data/labelled/inventory.csv")))
    by_photo = defaultdict(list)
    for row in csv.DictReader(open(path)):
        by_photo[(row.get("run", "1"), row["image"])].append(row)

    scores = {key: score_photo(preds, inventory) for key, preds in sorted(by_photo.items())}
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


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/predictions.csv")
