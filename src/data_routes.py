"""Data tab for the main app: browse the training images and their metadata. Read-only, never writes data.

Mounted in api.py with app.include_router(data_routes.router). Page at /data, JSON and thumbnails under /api/data.
Sources: data/train/index.csv (every training image), data/external/wild/verified.csv (OCR-verified wild photos,
shown as source "wild_verified"), plus the per-source products.csv files joined on product_key.
Images are served only for paths listed in the index, never arbitrary paths.
"""
import csv
import threading
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from datafiles import ROOT, WEB, jpeg, read_rows

INDEX = ROOT / "data" / "train" / "index.csv"
WILD = ROOT / "data" / "external" / "wild"
EXT = ROOT / "data" / "external"
RETAIL = ("saq", "systembolaget", "drinkfinder")
SIDES = {"thumb": 300, "full": 1000}
QUALITY = 82
# dropdown value -> display sources it covers
GROUPS = {"all": None, "off": {"off"}, "retail": set(RETAIL), "wild_verified": {"wild_verified"},
          "wild": {"wild"}, "beer_ru": {"beer_ru"}, "whiskey": {"whiskey"}, "stperez": {"stperez"},
          "beerbb": {"beerbb"}, "rp2k": {"rp2k"}, "ttb": {"ttb"}}
FIELDS = ("path", "source", "brand", "brand_key", "product", "product_key", "size", "group", "category", "gallery")

csv.field_size_limit(10**8)  # products.csv rows hold every image id or URL in one field
api = APIRouter(prefix="/api/data", tags=["data"])
router = APIRouter()
_lock = threading.Lock()
_state: dict | None = None


@router.get("/data", include_in_schema=False)
def data_page() -> FileResponse:
    return FileResponse(WEB / "data.html")


def state() -> dict:
    """Load every CSV once, on first use, and keep it in memory."""
    global _state
    with _lock:
        if _state is not None:
            return _state
        ocr = {f"data/external/wild/{r['file']}": r.get("text", "") for r in read_rows(WILD / "verified.csv")}
        rows = []
        for r in read_rows(INDEX):
            row = {k: r.get(k, "") for k in FIELDS}
            if row["source"] in ("wild", "wild_verified") and row["path"] in ocr:
                row["source"] = "wild_verified"
                row["ocr_text"] = ocr[row["path"]]
            row["id"] = len(rows)
            row["_q"] = f"{row['brand']} {row['product']}".lower()
            rows.append(row)
        products = {}
        for r in read_rows(EXT / "off_alcohol" / "products.csv"):
            r.pop("image_ids", None)
            products[f"off:{r.get('code', '')}"] = r
        for src in RETAIL:
            for r in read_rows(EXT / f"retail_{src}" / "products.csv"):
                r.pop("image_urls", None)
                products[f"{src}:{r.get('product_id', '')}"] = r
        for r in read_rows(EXT / "ttb-cola-spirits" / "index.csv"):
            products[f"ttb:{r.get('TTB ID', '')}"] = r
        by_product = {}
        for row in rows:
            if row["product_key"]:
                by_product.setdefault(row["product_key"], []).append(row["id"])
        _state = {"rows": rows, "paths": {r["path"]: r["id"] for r in rows},
                  "products": products, "by_product": by_product}
        return _state


def public(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def card(row: dict) -> dict:
    return {k: row[k] for k in ("id", "path", "brand", "product", "source", "size")}


@api.get("/items")
def items(source: str = "all", q: str = "", brand: str = "", page: int = 1, per: int = 60):
    """One page of the grid. Counts per source ignore the source filter so the dropdown can show them;
    the brand list ignores the brand filter so it stays a quick switcher."""
    if source not in GROUPS:
        raise HTTPException(400, f"unknown source {source}")
    per = max(1, min(per, 200))
    want, q, brand = GROUPS[source], q.strip().lower(), brand.strip().lower()
    base = [r for r in state()["rows"] if not q or q in r["_q"]]
    by_source = Counter(r["source"] for r in base if not brand or r["brand_key"] == brand)
    in_src = [r for r in base if want is None or r["source"] in want]
    brands = Counter((r["brand_key"], r["brand"]) for r in in_src if r["brand_key"])
    hits = [r for r in in_src if not brand or r["brand_key"] == brand]
    pages = max(1, -(-len(hits) // per))
    page = max(1, min(page, pages))
    return {"total": len(hits), "page": page, "pages": pages, "per": per,
            "sources": dict(by_source), "all": sum(by_source.values()),
            "brands": [{"key": k, "name": n, "count": c} for (k, n), c in brands.most_common(40)],
            "items": [card(r) for r in hits[(page - 1) * per: page * per]]}


@api.get("/item/{item_id}")
def item(item_id: int):
    s = state()
    if not 0 <= item_id < len(s["rows"]):
        raise HTTPException(404, "no such item")
    row = s["rows"][item_id]
    same = [s["rows"][i] for i in s["by_product"].get(row["product_key"], []) if i != item_id]
    brand_n = sum(r["brand_key"] == row["brand_key"] for r in s["rows"]) if row["brand_key"] else 0
    return {"item": public(row), "product_meta": s["products"].get(row["product_key"]),
            "same_product": [card(r) for r in same[:48]], "same_product_total": len(same),
            "brand_total": brand_n}


@api.get("/img")
def img(path: str, size: str = "thumb"):
    """JPEG of an indexed training image. Only paths present in the index are served; anything else is 404."""
    if size not in SIDES or path not in state()["paths"] or ".." in Path(path).parts:
        raise HTTPException(404, "not found")
    full = (ROOT / path).resolve()
    if not full.is_relative_to(ROOT / "data") or not full.is_file():
        raise HTTPException(404, "not found")
    try:
        data = jpeg(full, SIDES[size], QUALITY)
    except (OSError, ValueError) as e:
        raise HTTPException(404, "unreadable image") from e
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


router.include_router(api)
