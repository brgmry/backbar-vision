"""Steps shared by the scripts that name detected crops against the external catalogue (identify_trained.py,
compare_backbones.py, fuse.py): cut each crop again from the full-resolution photo, read a brand off its label,
keep the best crop per product, write the predictions CSV. The scripts differ only in how they score a crop.
"""
import csv
import math
import sys
from pathlib import Path

from PIL import Image
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/, for the shared helpers
from boxes import DETECT_SIDE, padded_crop, resized_scale  # noqa: E402
from identify import PRED_FIELDS  # noqa: E402
from photos import EXTS, open_photo  # noqa: E402
from schema import read_csv  # noqa: E402
from train_index import key  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CROP_PAD = 0.05  # widen each box a little so edge letters are not cut
MIN_BRAND_KEY = 5  # characters (spaces aside): shorter brand names turn up inside too much unrelated label text
BRAND_MATCH = 90  # fuzzy partial_ratio a brand name needs inside the label text
MIN_TEXT_SHARE = 0.6  # label text shorter than this share of a brand name can't hold it


def full_res_crops(photos_dir, crops_dir, out_dir: Path) -> tuple[list[dict], dict[str, Path], list[Path]]:
    """Each box in <crops_dir>/boxes.csv cut again from the full-resolution photo into out_dir/<k>.jpg: the saved
    crops come from the copy detect.py shrank. Returns the boxes, the photos by stem and the crop paths."""
    boxes = read_csv(Path(crops_dir) / "boxes.csv")
    photos = {p.stem: p for p in Path(photos_dir).iterdir() if p.suffix.lower() in EXTS}
    paths, name, photo = [], None, None
    for k, b in enumerate(boxes):
        if b["photo"] != name:
            name, photo = b["photo"], open_photo(photos[b["photo"]])
        sc = resized_scale(photo.size, DETECT_SIDE)
        x1, y1, x2, y2 = (float(b[c]) * sc for c in ("x1", "y1", "x2", "y2"))
        paths.append(out_dir / f"{k}.jpg")
        padded_crop(photo, [x1, y1, x2 - x1, y2 - y1], CROP_PAD)[0].save(paths[-1], quality=95)
    return boxes, photos, paths


def ocr_text(path: Path) -> str:
    """Label text of a crop (Apple Vision), normalised like the catalogue keys."""
    from ocr import read_text  # macOS only
    return key(" ".join(t for t, _ in read_text(path)))


def ocr_texts(crops_dir, boxes: list[dict], paths: list[Path]) -> list[str]:
    """ocr_text for every crop, cached in <crops_dir>/ocr_fullres.csv. The cache names its crops and is used only
    while boxes.csv lists the same ones in the same order, so re-running detection can't shift texts onto the
    wrong crops."""
    cache = Path(crops_dir) / "ocr_fullres.csv"
    want = [(b["photo"], b["crop"]) for b in boxes]
    if cache.exists():
        rows = read_csv(cache)
        if [(r.get("photo"), r.get("crop")) for r in rows] == want:
            return [r["text"] for r in rows]
        print(f"{cache} does not match {crops_dir}/boxes.csv: reading the labels again")
    texts = [ocr_text(p) for p in paths]
    with cache.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["photo", "crop", "text"])
        w.writerows([photo, crop, text] for (photo, crop), text in zip(want, texts, strict=True))
    return texts


def brand_names(rows: list[dict]) -> dict[str, str]:
    """brand_key -> display name, from the catalogue gallery and the top-brands list."""
    names = {r["brand_key"]: r["brand"] for r in rows if r["gallery"] == "1"}
    for r in read_csv(ROOT / "data/external/brands.csv"):
        names.setdefault(key(r["brand"]), r["brand"])
    return {k: v for k, v in names.items() if len(k.replace(" ", "")) >= MIN_BRAND_KEY}


def read_brand(text: str, names: dict[str, str]) -> tuple[str, str] | None:
    """Longest brand name found in the label text (fuzzy, to survive OCR slips like BACARD or JOUNNIE)."""
    if len(text) < 4:
        return None
    hits = [(k, v) for k, v in names.items()
            if fuzz.partial_ratio(k, text) >= BRAND_MATCH and len(text) >= len(k) * MIN_TEXT_SHARE]
    return max(hits, key=lambda kv: len(kv[0])) if hits else None


def brand_only(brand: tuple[str, str]) -> dict:
    """A catalogue-shaped row for a brand read off the label that has no product in the catalogue."""
    return {"brand": brand[1], "product": "", "product_key": f"brand:{brand[0]}", "group": "", "size": ""}


def keep_best(best: dict, photo: str, score: float, k: int, row: dict) -> None:
    """best[photo][product key] = (score, box index, row) of the highest-scoring crop: one answer per product."""
    if score > best[photo].get(row["product_key"], (-math.inf,))[0]:
        best[photo][row["product_key"]] = (score, k, row)


def write_predictions(path, boxes: list[dict], photos: dict[str, Path], best: dict, id_source: str,
                      label: str = "sim") -> int:
    """The kept crops as prediction rows, left to right in each photo, boxes in photo pixels. Returns the count."""
    out = []
    for photo, items in best.items():
        with Image.open(photos[photo]) as im:
            sc = resized_scale(im.size, DETECT_SIDE)
        for pos, (s, k, r) in enumerate(sorted(items.values(), key=lambda t: float(boxes[t[1]]["x1"])), 1):
            x1, y1, x2, y2 = (float(boxes[k][c]) * sc for c in ("x1", "y1", "x2", "y2"))
            out.append({"run": 1, "image": photos[photo].name, "position": pos,
                        "description": f"crop {boxes[k]['crop']}, {label} {s:.2f}",
                        "bbox": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)], "brand": r["brand"],
                        "product": r["product"], "category": r["group"], "liquid": "", "size": r["size"],
                        "unit": "", "glass": "", "fill_ml": None, "id_source": id_source, "fill_visible": False})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        w.writeheader()
        w.writerows(out)
    return len(out)
