"""Move wild training photos within pHash distance holdout.MAX_PHASH_DIST of a bar test photo out of the data.

Uses the pHash each photo got when it was collected. Moved photos go to data/external/wild/_test_overlap/ and are
listed in its dropped.csv; the per-brand manifests, manifest.csv and verified.csv are rewritten without them.
Stops if data/test_bars/phashes.txt is missing or empty. train_index.py checks every source again.
usage: python src/experiments/drop_test_overlap.py
"""
import csv
import json
import shutil

import imagehash

from collect_wild import OUT as WILD
from collect_wild import merge
from holdout import MAX_PHASH_DIST, test_distance, test_hashes

HELD = WILD / "_test_overlap"  # no _manifest.json at the top, so the collectors' globs skip it


def drop(test) -> list[dict]:
    dropped = []
    for m in sorted(WILD.glob("*/_manifest.json")):
        with open(m) as f:
            rows = json.load(f)
        keep = []
        for r in rows:
            dist = test_distance(imagehash.hex_to_hash(r["phash"]), test)
            if dist > MAX_PHASH_DIST:
                keep.append(r)
                continue
            if (WILD / r["file"]).exists():
                (HELD / r["file"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(WILD / r["file"], HELD / r["file"])
            dropped.append({"file": r["file"], "brand": r["brand"], "phash": r["phash"], "distance": dist})
        if len(keep) < len(rows):
            with open(m, "w") as f:
                json.dump(keep, f)
    return dropped


def record(dropped: list[dict]) -> None:
    """Append to _test_overlap/dropped.csv, so every run's drops stay listed."""
    path = HELD / "dropped.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "brand", "phash", "distance"])
        if new:
            w.writeheader()
        w.writerows(dropped)


def prune_verified(files: set[str]) -> None:
    """verified.csv feeds gallery v2 and eval_synth directly, so it must not list moved photos either."""
    path = WILD / "verified.csv"
    if not files or not path.exists():
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(r for r in rows if r["file"] not in files)


def main():
    test = test_hashes()
    print("test hashes", len(test))
    dropped = drop(test)
    record(dropped)
    prune_verified({d["file"] for d in dropped})
    print("dropped", len(dropped), f"(moved to {HELD})" if dropped else "")
    merge()


if __name__ == "__main__":
    main()
