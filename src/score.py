"""Score predictions against the inventory of the kitchen photo set.

Every photo shows every inventory item. Per photo, predicted bottles are matched one to one to inventory items by
fuzzy brand + product similarity (Hungarian assignment). A photo of data/raw with no prediction rows scores zero.

With --alcohol only alcohol items count, in the inventory and in the predictions. If data/labelled/fill.csv exists
(image, id, fill_ml: measured, never model-made; a blank image means every photo), it also reports the fill error
in percentage points of bottle size, over matched bottles whose fill was visible.

Usage: python src/score.py [data/predictions.csv] [--alcohol]
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from photos import EXTS
from schema import ALCOHOL, BAR, size_ml

BRAND_THRESHOLD = 80  # brand_recall: the brands must match, not just product words
MATCH_THRESHOLD = 60  # token_set_ratio below this counts as a wrong identification
SIZE_TOL = 0.1  # a predicted size within 10% of the true size counts as right
ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "data/labelled/inventory.csv"
FILL_TRUTH = ROOT / "data/labelled/fill.csv"
PHOTOS = ROOT / "data/raw"


def read_csv(path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def norm(row: dict) -> str:
    return f"{row['brand']} {row['product']}".lower()


def brand_ok(pred: dict, item: dict) -> bool:
    """Strict: the predicted brand names the item's brand. Generic names ("Coffee Liqueur") and "Unknown" fail."""
    p, i = (pred["brand"] or "").lower().strip(), item["brand"].lower().strip()
    return bool(p and i and p != "unknown" and fuzz.token_set_ratio(p, i) >= BRAND_THRESHOLD)


def size_ok(pred: dict, item: dict) -> bool:
    p, i = size_ml(pred), size_ml(item)
    return p is not None and bool(i) and abs(p - i) / i <= SIZE_TOL


def match_photo(preds: list[dict], inventory: list[dict]) -> list[dict | None]:
    """For each prediction, the inventory row it is matched to one to one, or None."""
    if not preds:
        return []
    sim = np.array([[fuzz.token_set_ratio(norm(p), norm(i)) for i in inventory] for p in preds])
    out = [None] * len(preds)
    for r, c in zip(*linear_sum_assignment(-sim), strict=True):
        if sim[r, c] >= MATCH_THRESHOLD:
            out[r] = inventory[c]
    return out


def matched_pairs(preds: list[dict], inventory: list[dict]) -> list[tuple[dict, dict]]:
    return [(p, i) for p, i in zip(preds, match_photo(preds, inventory), strict=True) if i is not None]


def score_matched(preds: list[dict], inventory: list[dict], matched: list[tuple[dict, dict]]) -> dict[str, float]:
    n = len(inventory)
    bar = [i for i in inventory if i["category"] in BAR]
    branded = sum(bool(i["brand"]) for i in inventory)
    return {
        "count_error": len(preds) - n,
        "recall": len(matched) / n,
        "precision": len(matched) / len(preds) if preds else 0.0,
        "category_acc": sum(p["category"] == i["category"] for p, i in matched) / n,
        "size_acc": sum(size_ok(p, i) for p, i in matched) / n,
        # strict: matched AND the brand is right; items with no known brand are left out
        "brand_recall": sum(brand_ok(p, i) for p, i in matched) / max(1, branded),
        # bar items only (spirits, liqueurs, wine, beer, mixers); other predictions are ignored
        "bar_recall": sum(i["category"] in BAR for _, i in matched) / len(bar) if bar else 0.0,
    }


def score_photo(preds: list[dict], inventory: list[dict]) -> dict[str, float]:
    return score_matched(preds, inventory, matched_pairs(preds, inventory))


def fill_truth(truth: dict, image: str, item_id: str) -> float | None:
    return truth.get((image, item_id), truth.get(("", item_id)))


def fill_errors(matched: list[tuple[dict, dict]], truth: dict) -> list[float]:
    """Absolute fill error in percentage points of bottle size, for matched bottles with a measured fill."""
    return [abs(float(p["fill_ml"]) - fill_truth(truth, p["image"], i["id"])) / size_ml(i) * 100
            for p, i in matched
            if fill_truth(truth, p["image"], i["id"]) is not None and p.get("fill_ml") not in (None, "")]


def main(path: str, alcohol: bool = False) -> None:
    inventory = read_csv(INVENTORY)
    if alcohol:
        inventory = [i for i in inventory if i["category"] in ALCOHOL]
    by_photo = defaultdict(list)
    for row in read_csv(path):
        if not alcohol or row["category"] in ALCOHOL:
            by_photo[(row.get("run") or "1", row["image"])].append(row)
    kitchen = {p.name for p in PHOTOS.iterdir() if p.suffix.lower() in EXTS}
    if kitchen & {image for _, image in by_photo}:  # a kitchen-set run: photos with no predictions count as misses
        for run in {run for run, _ in by_photo}:
            for image in kitchen:
                by_photo.setdefault((run, image), [])

    truth = {(r.get("image", ""), r["id"]): float(r["fill_ml"]) for r in read_csv(FILL_TRUTH)} if FILL_TRUTH.exists() else {}
    scores, errors, measured = {}, [], 0
    for key, preds in sorted(by_photo.items()):
        matched = matched_pairs(preds, inventory)
        scores[key] = score_matched(preds, inventory, matched)
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
        print(f"  fill MAE      {mae:>7} over {len(errors)}/{measured} matched bottles with a measured fill "
              "(the rest: fill not visible)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions", nargs="?", default=str(ROOT / "data/predictions.csv"))
    ap.add_argument("--alcohol", action="store_true", help="score alcohol items only")
    args = ap.parse_args()
    main(args.predictions, args.alcohol)
