"""Testing tab for the main app: leaderboard, Claude runs, photo inspector. Read-only, never calls a model.

Mounted in api.py with app.include_router(testing_routes.router). Page at /testing, JSON under /api/test, and
/common.js, the script every page loads. File readers and run scoring come from src/datafiles.py.
"""
import statistics
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

import datafiles as df
import judge_routes
import score

DATA = df.DATA
FLOATS = ("recall", "recall_sd", "brand_recall", "bar_recall", "precision", "count_error")
VIEW_SIDE = 1200

api = APIRouter(prefix="/api/test", tags=["testing"])
router = APIRouter()
router.include_router(df.router)  # /common.js; api.py only mounts the three tab routers


@router.get("/testing", include_in_schema=False)
def testing_page() -> FileResponse:
    return FileResponse(df.WEB / "testing.html")


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def truth_boxes() -> dict[str, list[tuple[int, list[float]]]]:
    """Per data/raw photo: (item_id, [x, y, w, h]) of the largest hand-labelled detector crop of each item,
    in original-photo pixels. The only ground-truth boxes we have."""
    labels = {(r["photo"], r["crop"]): r["item_id"] for r in df.read_csv(DATA / "crops" / "labels.csv")}
    scales = {p.stem: df.detector_scale(df.image_size(p)) for p in df.kitchen_photos()}
    best = defaultdict(dict)
    for b in df.read_csv(DATA / "crops" / "boxes.csv"):
        item = labels.get((b["photo"], b["crop"]), "")
        if b["photo"] not in scales or not item.isdigit() or int(item) <= 0:
            continue
        try:
            x1, y1, x2, y2 = (float(b[k]) * scales[b["photo"]] for k in ("x1", "y1", "x2", "y2"))
        except ValueError:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area > best[b["photo"]].get(int(item), (0,))[0]:
            best[b["photo"]][int(item)] = (area, [x1, y1, x2 - x1, y2 - y1])
    return {photo: [(i, box) for i, (_, box) in items.items()] for photo, items in best.items()}


def inside(point, box) -> bool:
    return box[0] <= point[0] <= box[0] + box[2] and box[1] <= point[1] <= box[1] + box[3]


def bottles_found(f: Path, truth) -> float | None:
    """Share of ground-truth items whose box centre falls inside any predicted box, mean per run then across
    runs. Only for data/raw photos with predicted bboxes; None when that can't be computed."""
    runs = defaultdict(lambda: defaultdict(list))
    for r in df.read_csv(f):
        stem = Path(r.get("image", "")).stem
        box = df.parse_bbox(r.get("bbox", ""))
        if stem in truth and box:
            runs[r.get("run", "1")][stem].append(box)
    if not runs:
        return None
    means = []
    for photos in runs.values():
        per = []
        for stem, boxes in photos.items():
            items = truth[stem]
            hit = sum(any(inside((x + w / 2, y + h / 2), b) for b in boxes) for _, (x, y, w, h) in items)
            per.append(hit / len(items))
        means.append(statistics.mean(per))
    return statistics.mean(means)


def brand_recall_key(row: dict) -> tuple:
    return row["brand_recall"] is None, -(row["brand_recall"] or 0)


@api.get("/leaderboard")
def leaderboard():
    """data/results.csv, numbers parsed, plus bottles_found computed from the predictions file where possible."""
    truth = truth_boxes()
    out = []
    for r in df.read_csv(DATA / "results.csv"):
        row = {k: r.get(k, "") for k in ("date", "experiment", "model", "predictions_file", "inputs", "notes")}
        row |= {k: _num(r.get(k)) for k in FLOATS}
        row["valid"] = r.get("follows_rule") == "yes"
        row["baseline"] = r.get("experiment", "").lower().startswith("claude baseline")
        f = DATA / Path(r.get("predictions_file", "")).name
        row["bottles_found"] = bottles_found(f, truth) if f.is_file() and f.suffix == ".csv" else None
        out.append(row)
    out += judge_routes.leaderboard_rows()  # human-judged bar accuracy, live from data/test_bars/judgements.csv
    return {"valid": sorted((r for r in out if r["valid"]), key=brand_recall_key),
            "invalid": sorted((r for r in out if not r["valid"]), key=brand_recall_key)}


@api.get("/runs")
def runs(file: str = "predictions.csv"):
    """Per-run means and per-photo rows of one predictions file, scored against the inventory."""
    f = DATA / Path(file).name
    if not f.is_file():
        raise HTTPException(404, f"no predictions file {file}")
    return {"file": f.name, "runs": df.scored_runs(f)}


def find_photo(name: str) -> tuple[str, Path]:
    name = Path(name).name
    for p in df.kitchen_photos():
        if name in (p.name, p.stem):
            return "raw", p
    p = df.BARS_IMG / name
    if df.BAR_NAME.match(name) and p.is_file():
        return "bars", p
    raise HTTPException(404, f"no photo {name}")


@api.get("/photos")
def photos():
    return ([{"name": p.name, "set": "raw"} for p in df.kitchen_photos()]
            + [{"name": p.name, "set": "bars"} for p in df.bar_photos()])


@api.get("/files")
def files():
    return [f.name for f in df.prediction_files()]


@api.get("/photo/{name}")
def photo(name: str):
    _, p = find_photo(name)
    return Response(df.jpeg(p, VIEW_SIDE), media_type="image/jpeg")


@api.get("/inspect")
def inspect(photo: str, file: str):
    """One photo + one predictions file: every run's rows with boxes, and for data/raw photos the ground truth
    and which inventory item each row was matched to."""
    kind, p = find_photo(photo)
    f = DATA / Path(file).name
    if not f.is_file():
        raise HTTPException(404, f"no predictions file {file}")
    w, h = df.image_size(p)
    inv = df.inventory() if kind == "raw" else []
    by_run = defaultdict(list)
    for r in df.read_csv(f):
        if r.get("image") in (p.name, p.stem):
            by_run[r.get("run", "1")].append(r)
    out = {}
    for run, rows in sorted(by_run.items(), key=lambda kv: df.run_key(kv[0])):
        matches = score.match_photo(rows, inv) if inv else [None] * len(rows)
        out[run] = [{k: r.get(k, "") for k in ("position", "brand", "product", "category", "size", "unit", "fill_ml",
                                               "id_source")}
                    | {"bbox": df.parse_bbox(r.get("bbox", "")),
                       "matched": f"{m['brand']} {m['product']}".strip() if m else None,
                       "brand_ok": bool(m and score.brand_ok(r, m))}
                    for r, m in zip(rows, matches, strict=True)]
    truth = [{"id": i["id"], "brand": i["brand"], "product": i["product"], "category": i["category"]} for i in inv]
    return {"photo": p.name, "set": kind, "width": w, "height": h, "runs": out, "truth": truth}


router.include_router(api)
