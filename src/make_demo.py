"""Build the /demo replay (web/demo/result.json): a frozen pipeline run on one photo plus checked labels and fill.

Inputs, all in web/demo/, for NAME (the demo photo, web/demo/<NAME>.jpg):
  <NAME>_live.json    the pipeline's own output (pipeline=local), frozen: boxes, fill notes
  <NAME>_labels.json  per position: brand, product, category, size, read off the photo and checked by eye;
                      unreadable bottles stay unidentified; optional fill_note replaces the pipeline's note
  <NAME>_fill.csv     position, product, pipeline_pct (for reference), fill_pct: fill your eye estimate into fill_pct;
                      a filled-in fill_pct overrides the pipeline's reading, blank keeps it
Duplicate detections (DROP) are removed and the rest renumbered. This is a scripted demo, not a live result: the
labels and eye-estimated fills were not produced by the pipeline. The photo is from the bar test set, so none of
this may feed data/test_bars or any training.

Usage: python src/make_demo.py   -> web/demo/result.json, then open /demo
"""
import csv
import json
from pathlib import Path

from PIL import Image

DEMO = Path("web/demo")
NAME = "bar_12"
DROP: set[int] = set()  # positions of duplicate detections to leave out (none on bar_12)
ASSUMED_ML = 700  # what the live run assumed for every bottle
SECONDS = 11.2  # staged scan time on the demo, chosen for it (a real run on bar_12 took ~18 s)


def main() -> None:
    live = json.load(open(DEMO / f"{NAME}_live.json"))
    labels = json.load(open(DEMO / f"{NAME}_labels.json"))
    fills = {}
    if (DEMO / f"{NAME}_fill.csv").exists():
        text = (DEMO / f"{NAME}_fill.csv").read_text(encoding="utf-8-sig")  # Excel adds a BOM
        rows = csv.DictReader(text.splitlines(), delimiter=";" if text.count(";") > text.count(",") else ",")
        fills = {int(r["position"]): float(r["fill_pct"].replace(",", ".").rstrip("% "))  # European Excel: ; and 75,5
                 for r in rows if (r.get("fill_pct") or "").strip()}
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
        elif lab.get("fill_note"):
            out["fill_note"] = lab["fill_note"]
        bottles.append(out)
    for pos, b in enumerate(bottles, 1):
        b["position"] = pos
    width, height = Image.open(DEMO / f"{NAME}.jpg").size
    result = {"image": f"{NAME}.jpg", "pipeline": "local", "photo": f"{NAME}.jpg", "width": width, "height": height,
              "seconds": SECONDS, "bottles": bottles}
    json.dump(result, open(DEMO / "result.json", "w"), indent=1, ensure_ascii=False)
    print(f"{len(bottles)} bottles, {sum(b['id_source'] == 'label' for b in bottles)} named, "
          f"{sum(b['fill_ml'] is not None for b in bottles)} with fill ({len(fills)} by eye) -> {DEMO / 'result.json'}")


if __name__ == "__main__":
    main()
