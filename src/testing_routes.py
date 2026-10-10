"""Testing tab for the main app: leaderboard, Claude runs, photo inspector. Read-only, never calls a model.

Mounted in api.py with app.include_router(testing_routes.router). Page at /testing, JSON under /api/test.
Reuses src/dashboard.py (file readers, run scoring) and src/score.py (matching) by import.
"""
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

import dashboard as dash  # registers the HEIC opener; its own FastAPI app is never served from here

WEB = Path(__file__).resolve().parent.parent / "web"
DATA = dash.DATA
FLOATS = ("recall", "recall_sd", "brand_recall", "bar_recall", "precision", "count_error")

api = APIRouter(prefix="/api/test", tags=["testing"])
router = APIRouter()


@router.get("/testing", include_in_schema=False)
def testing_page() -> FileResponse:
    return FileResponse(WEB / "testing.html")


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=64)
def _size(path: str, mtime: float) -> tuple[int, int]:
    return Image.open(path).size


def size(p: Path) -> tuple[int, int]:
    return _size(str(p), p.stat().st_mtime)


def truth_boxes() -> dict[str, list[tuple[int, list[float]]]]:
    """Per data/raw photo: (item_id, [x, y, w, h]) of the largest hand-labelled detector crop of each item,
    in original-photo pixels. The only ground-truth boxes we have."""
    labels = {(r["photo"], r["crop"]): r["item_id"] for r in dash.read_csv(DATA / "crops" / "labels.csv")}
    sizes = {p.stem: size(p) for p in dash.photos()}
    best = defaultdict(dict)
    for b in dash.read_csv(DATA / "crops" / "boxes.csv"):
        item = labels.get((b["photo"], b["crop"]), "")
        if b["photo"] not in sizes or not item.isdigit() or int(item) <= 0:
            continue
        s = max(sizes[b["photo"]]) / dash.DETECT_SIDE
        try:
            x1, y1, x2, y2 = (float(b[k]) * s for k in ("x1", "y1", "x2", "y2"))
        except ValueError:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area > best[b["photo"]].get(int(item), (0,))[0]:
            best[b["photo"]][int(item)] = (area, [x1, y1, x2 - x1, y2 - y1])
    return {photo: [(i, box) for i, (_, box) in items.items()] for photo, items in best.items()}


def bottles_found(f: Path, truth) -> float | None:
    """Share of ground-truth items whose box centre falls inside any predicted box, mean per run then across
    runs. Only for data/raw photos with predicted bboxes; None when that can't be computed."""
    runs = defaultdict(lambda: defaultdict(list))
    for r in dash.read_csv(f):
        stem = Path(r.get("image", "")).stem
        box = dash.parse_bbox(r.get("bbox", ""))
        if stem in truth and box:
            runs[r.get("run", "1")][stem].append(box)
    if not runs:
        return None
    inside = lambda c, b: b[0] <= c[0] <= b[0] + b[2] and b[1] <= c[1] <= b[1] + b[3]
    means = []
    for photos in runs.values():
        per = []
        for stem, boxes in photos.items():
            items = truth[stem]
            hit = sum(any(inside((x + w / 2, y + h / 2), b) for b in boxes) for _, (x, y, w, h) in items)
            per.append(hit / len(items))
        means.append(statistics.mean(per))
    return statistics.mean(means)


@api.get("/leaderboard")
def leaderboard():
    """data/results.csv, numbers parsed, plus bottles_found computed from the predictions file where possible."""
    truth = truth_boxes()
    out = []
    for r in dash.read_csv(DATA / "results.csv"):
        row = {k: r.get(k, "") for k in ("date", "experiment", "model", "predictions_file", "inputs", "notes")}
        row |= {k: _num(r.get(k)) for k in FLOATS}
        row["valid"] = r.get("follows_rule") == "yes"
        row["baseline"] = r.get("experiment", "").lower().startswith("claude baseline")
        f = DATA / Path(r.get("predictions_file", "")).name
        row["bottles_found"] = bottles_found(f, truth) if f.is_file() and f.suffix == ".csv" else None
        out.append(row)
    key = lambda x: (x["brand_recall"] is None, -(x["brand_recall"] or 0))
    return {"valid": sorted((r for r in out if r["valid"]), key=key),
            "invalid": sorted((r for r in out if not r["valid"]), key=key)}


@api.get("/runs")
def runs(file: str = "predictions.csv"):
    """Per-run means and per-photo rows of one predictions file, scored against the inventory (dashboard logic)."""
    if dash.scorer is None:
        raise HTTPException(503, "src/score.py could not be imported")
    return dash.api_runs(Path(file).name)


def find_photo(name: str) -> tuple[str, Path]:
    name = Path(name).name
    for p in dash.photos():
        if name in (p.name, p.stem):
            return "raw", p
    p = dash.BARS_IMG / name
    if dash.BAR_NAME.match(name) and p.is_file():
        return "bars", p
    raise HTTPException(404, f"no photo {name}")


@api.get("/photos")
def photos():
    raw = [{"name": p.name, "set": "raw"} for p in dash.photos()]
    bars = [{"name": p.name, "set": "bars"} for p in sorted(dash.BARS_IMG.glob("bar_*.jpg"))] if dash.BARS_IMG.exists() else []
    return raw + bars


@api.get("/files")
def files():
    return [f.name for f in dash.prediction_files()]


@api.get("/photo/{name}")
def photo(name: str):
    kind, p = find_photo(name)
    return dash.photo_jpeg(p.name) if kind == "raw" else dash.bar_jpeg(p.name, side=dash.VIEW_SIDE)


@api.get("/inspect")
def inspect(photo: str, file: str):
    """One photo + one predictions file: every run's rows with boxes, and for data/raw photos the ground truth
    and which inventory item each row was matched to."""
    kind, p = find_photo(photo)
    f = DATA / Path(file).name
    if not f.is_file():
        raise HTTPException(404, f"no predictions file {file}")
    w, h = size(p)
    inv = dash.inventory() if kind == "raw" else []
    by_run = defaultdict(list)
    for r in dash.read_csv(f):
        if r.get("image") in (p.name, p.stem):
            by_run[r.get("run", "1")].append(r)
    out = {}
    for run, rows in sorted(by_run.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
        matches = dash.scorer.match_photo(rows, inv) if inv and dash.scorer else [None] * len(rows)
        out[run] = [{k: r.get(k, "") for k in ("position", "brand", "product", "category", "size", "unit", "fill_ml", "id_source")}
                    | {"bbox": dash.parse_bbox(r.get("bbox", "")),
                       "matched": f"{m['brand']} {m['product']}".strip() if m else None,
                       "brand_ok": bool(m and dash.scorer.brand_ok(r, m))}
                    for r, m in zip(rows, matches)]
    truth = [{"id": i["id"], "brand": i["brand"], "product": i["product"], "category": i["category"]} for i in inv]
    return {"photo": p.name, "set": kind, "width": w, "height": h, "runs": out, "truth": truth}


router.include_router(api)
