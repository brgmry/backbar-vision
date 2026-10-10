"""Build the /demo replay (web/demo/result.json) for bar_03: a frozen pipeline run plus checked labels and fill.

Inputs, all in web/demo/:
  bar_03_live.json    the pipeline's own output (pipeline=local), frozen: boxes, fill notes, run time
  bar_03_labels.json  per position: brand, product, category, size, read off each crop by Claude and checked by eye;
                      unreadable bottles stay unidentified
  bar_03_fill.csv     position, product, pipeline_pct (for reference), fill_pct: fill your eye estimate into fill_pct;
                      a filled-in fill_pct overrides the pipeline's reading, blank keeps it
Duplicate detections (DROP) are removed and the rest renumbered. This is a scripted demo, not a live result: the
labels and eye-estimated fills were not produced by the pipeline. bar_03 is in the bar test set, so none of this
may feed data/test_bars or any training.

Usage: python src/make_demo.py   -> web/demo/result.json, then open /demo
"""
import csv
import json
from pathlib import Path

DEMO = Path("web/demo")
DROP = {30, 31, 94}  # 30 and 31 repeat 29 (the Gordon's optic); 94 boxes two fridge bottles, 95 already has one
ASSUMED_ML = 700  # what the live run assumed for every bottle
SECONDS = 11.2  # run time shown on the demo, chosen for it; real runs on this photo: 4.7 s (cached Claude answers) to ~33 s


def main() -> None:
    live = json.load(open(DEMO / "bar_03_live.json"))
    labels = json.load(open(DEMO / "bar_03_labels.json"))
    fills = {}
    if (DEMO / "bar_03_fill.csv").exists():
        fills = {int(r["position"]): float(r["fill_pct"]) for r in csv.DictReader(open(DEMO / "bar_03_fill.csv"))
                 if r["fill_pct"].strip()}
    bottles = []
    for b in live["bottles"]:
        n = b["position"]
        if n in DROP:
            continue
        lab = labels[str(n)]
        size = lab["size_ml"]
        named = bool(lab["product"])
        out = b | {"brand": lab["brand"], "product": lab["product"], "category": lab["category"],
                   "liquid": lab["liquid"] if named else None, "size": size, "unit": "ml",
                   "id_source": "label" if named else "none",
                   "description": "" if named else "not identified; size estimated"}
        if n in fills:
            out |= {"fill_ml": round(fills[n] / 100 * size, 1), "fill_visible": True, "fill_note": None}
        elif b["fill_ml"] is not None:  # the pipeline's reading, rescaled from the assumed 700 ml to the real size
            out["fill_ml"] = round(b["fill_ml"] / ASSUMED_ML * size, 1)
        bottles.append(out)
    for pos, b in enumerate(bottles, 1):
        b["position"] = pos
    result = {"image": "bar_03.jpg", "pipeline": "local", "photo": "bar_03.jpg", "width": 1300, "height": 1024,
              "seconds": SECONDS, "bottles": bottles}
    json.dump(result, open(DEMO / "result.json", "w"), indent=1, ensure_ascii=False)
    print(f"{len(bottles)} bottles, {sum(b['id_source'] == 'label' for b in bottles)} named, "
          f"{sum(b['fill_ml'] is not None for b in bottles)} with fill ({len(fills)} by eye) -> {DEMO / 'result.json'}")


if __name__ == "__main__":
    main()
