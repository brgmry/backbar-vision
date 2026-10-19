"""Score predictions on the bar test set against the human labels in data/test_bars/labels/.

Each label file lists the bottles a person could identify in one photo, one per line as `brand | product | size`
(size optional), written on the Label bars page of the dashboard. Only labelled photos count. Bottles the labeller
could not identify are not in the file, so precision and count error would punish right answers and are not
reported; item recall, brand recall and size accuracy are.

Usage: python src/score_bars.py [data/predictions_bars_claude.csv] [--tier easy]
"""
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from schema import read_csv
from score import score_photo

ROOT = Path(__file__).resolve().parent.parent
BARS = ROOT / "data/test_bars"
LABELS = BARS / "labels"
METRICS = ["recall", "brand_recall", "size_acc"]


def parse_label(text: str) -> list[dict]:
    """Inventory rows from a label file: `brand | product | size` per line, blank lines skipped."""
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")] + ["", ""]
        rows.append({"id": str(n), "brand": parts[0], "product": parts[1], "size": parts[2], "unit": "",
                     "category": ""})
    return rows


def load_labels(tier: str | None = None) -> dict[str, list[dict]]:
    """Labelled bottles per photo name, for photos with at least one bottle, optionally one tier only."""
    tiers = {r["file"]: r["tier"] for r in read_csv(BARS / "manifest.csv")}
    labels = {}
    for path in sorted(LABELS.glob("*.txt")) if LABELS.exists() else []:
        image = path.stem + ".jpg"
        if tier and tiers.get(image) != tier:
            continue
        rows = parse_label(path.read_text())
        if rows:
            labels[image] = rows
    return labels


def main(path: str, tier: str | None = None) -> None:
    labels = load_labels(tier)
    if not labels:
        raise SystemExit(f"No labelled photos in {LABELS}" + (f" for tier {tier}" if tier else ""))
    preds = defaultdict(list)
    for row in read_csv(path):
        preds[(row.get("run") or "1", row["image"])].append(row)
    runs = sorted({run for run, _ in preds} or {"1"})

    print(f"{'run':>3} {'image':<12}{'bottles':>8}" + "".join(f"{m:>14}" for m in METRICS))
    run_means = defaultdict(list)
    for run in runs:
        scores = []
        for image, inventory in labels.items():
            s = score_photo(preds.get((run, image), []), inventory)
            scores.append(s)
            print(f"{run:>3} {image:<12}{len(inventory):>8}" + "".join(f"{s[m]:>14.2f}" for m in METRICS))
        for m in METRICS:
            run_means[m].append(statistics.mean(s[m] for s in scores))

    bottles = sum(len(v) for v in labels.values())
    print(f"\n{len(runs)} runs x {len(labels)} labelled photos, {bottles} labelled bottles")
    for m in METRICS:
        vals = run_means[m]
        spread = f" ± {statistics.stdev(vals):.3f} across runs" if len(vals) > 1 else ""
        print(f"  {m:<14}{statistics.mean(vals):>7.3f}{spread}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("predictions", nargs="?", default=str(ROOT / "data/predictions_bars_claude.csv"))
    ap.add_argument("--tier", choices=["easy", "medium", "hard"], help="score one tier of bar_35 to bar_82 only")
    args = ap.parse_args()
    main(args.predictions, args.tier)
