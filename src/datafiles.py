"""File readers and image helpers shared by the dashboard (dashboard.py) and the Testing, Data and Judge tabs.

Read-only. CSVs and thumbnails are cached by file modification time, so a file that is still being written shows
up on the next request without reading every file again on every request.
"""
import csv
import io
import json
import re
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

import score
from boxes import resized_scale
from photos import EXTS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WEB = ROOT / "web"
RAW = DATA / "raw"
BARS_IMG = DATA / "test_bars" / "img"
BAR_NAME = re.compile(r"^bar_\d{2,3}\.jpg$")
ORIENTATION = 0x0112  # EXIF tag
TURNED = {5, 6, 7, 8}  # orientations that swap width and height

router = APIRouter()


@router.get("/common.js", include_in_schema=False)
def common_js() -> FileResponse:
    """The helpers every page loads. A fixed path: nothing from the request picks the file."""
    return FileResponse(WEB / "common.js", media_type="text/javascript")


def read_rows(path: Path) -> list[dict]:
    """Rows of a CSV, tolerating a missing file or a half-written last line."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return [r for r in csv.DictReader(f) if r and None not in r and None not in r.values()]
    except (FileNotFoundError, csv.Error, UnicodeDecodeError):
        return []


def read_csv(path: Path) -> list[dict]:
    """read_rows, cached until the file changes. Each call gets its own copies, so callers may edit them."""
    try:
        st = path.stat()
    except OSError:
        return []
    return [dict(r) for r in _read_cached(str(path), st.st_mtime_ns, st.st_size)]


@lru_cache(maxsize=64)
def _read_cached(path: str, mtime_ns: int, size: int) -> tuple[dict, ...]:
    return tuple(read_rows(Path(path)))


def kitchen_photos() -> list[Path]:
    return sorted(p for p in RAW.iterdir() if p.suffix.lower() in EXTS) if RAW.exists() else []


def kitchen_photo(name: str) -> Path:
    for p in kitchen_photos():
        if name in (p.name, p.stem):
            return p
    raise HTTPException(404, f"no photo {name}")


def bar_photos() -> list[Path]:
    return sorted(BARS_IMG.glob("bar_*.jpg")) if BARS_IMG.exists() else []


def bar_photo(name: str) -> Path:
    """Only bar_NN.jpg files that actually exist in data/test_bars/img."""
    if not BAR_NAME.match(name) or not (BARS_IMG / name).is_file():
        raise HTTPException(404, f"no bar photo {name}")
    return BARS_IMG / name


def inventory() -> list[dict]:
    return read_csv(DATA / "labelled" / "inventory.csv")


def prediction_files() -> list[Path]:
    return sorted(DATA.glob("predictions*.csv"))


def parse_bbox(s: str) -> list[float] | None:
    try:
        b = json.loads(s)
        return [float(v) for v in b] if isinstance(b, list) and len(b) == 4 else None
    except (ValueError, TypeError):
        return None


def run_key(run: str) -> tuple:
    """Sort key for run labels: numbered runs in order, then any others by name."""
    return (0, int(run), "") if run.isdigit() else (1, 0, run)


def detector_scale(size: tuple[int, int]) -> float:
    """boxes.csv is in the photo detect.py shrank: this scales its boxes back to photo pixels."""
    from detect import MAX_SIDE  # imports torch, so only once a page needs detector boxes
    return resized_scale(size, MAX_SIDE)


def image_size(path: Path) -> tuple[int, int]:
    """(width, height) as shown, after EXIF orientation."""
    return _image_size(str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=512)
def _image_size(path: str, mtime_ns: int) -> tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
        return (h, w) if im.getexif().get(ORIENTATION) in TURNED else (w, h)


def jpeg(path: Path, side: int, quality: int = 85) -> bytes:
    """The image upright and shrunk to fit side x side, as JPEG, transparency on white. Raises OSError or
    ValueError when the file is not a readable image."""
    return _jpeg(str(path), path.stat().st_mtime_ns, side, quality)


@lru_cache(maxsize=256)  # thumbnails are 20-300 KB: a bounded cache stays well under 100 MB
def _jpeg(path: str, mtime_ns: int, side: int, quality: int) -> bytes:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
        im = im.convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return buf.getvalue()


def scored_runs(f: Path) -> list[dict]:
    """Every run of one predictions file scored against the kitchen inventory (score.py): per-run means, per-photo
    scores, and each row with the item it matched."""
    inv = inventory()
    names = [f"{i['brand']} {i['product']}".strip() for i in inv]
    by_run = defaultdict(lambda: defaultdict(list))
    for r in read_csv(f):
        if r.get("image"):
            by_run[r.get("run", "1")][r["image"]].append(r)
    runs = []
    for run, photos in sorted(by_run.items(), key=lambda kv: run_key(kv[0])):
        per_photo = []
        for image, rows in sorted(photos.items()):
            matches = score.match_photo(rows, inv)
            for r, m in zip(rows, matches, strict=True):
                r["matched"] = f"{m['brand']} {m['product']}".strip() if m else ""
            pairs = [(r, m) for r, m in zip(rows, matches, strict=True) if m is not None]
            found = {r["matched"] for r in rows if r["matched"]}
            per_photo.append({"image": image, "scores": score.score_matched(rows, inv, pairs), "rows": rows,
                              "missed": [n for n in names if n not in found]})
        metrics = list(per_photo[0]["scores"])
        runs.append({"run": run, "means": {m: statistics.mean(p["scores"][m] for p in per_photo) for m in metrics},
                     "photos": per_photo})
    return runs
