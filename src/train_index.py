"""Build the training index from external data only: one row per image with brand and product keys.

Sources: Open Food Facts alcohol + mixers, retailer catalogues, in-the-wild brand photos, Russian beer set.
Usage: python src/train_index.py   -> data/train/index.csv
"""
import csv
import re
import unicodedata
from pathlib import Path

EXT = Path("data/external")
OUT = Path("data/train")
IMG = {".jpg", ".jpeg", ".png", ".webp"}


def key(s: str) -> str:
    """Normalised brand/product key: ascii, lowercase, alphanumerics only."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def first_brand(brands: str) -> str:
    return (brands or "").split(",")[0].strip()


def rows():
    # Open Food Facts: product = barcode, several views per product
    for p in csv.DictReader(open(EXT / "off_alcohol/products.csv")):
        brand = first_brand(p["brands"])
        d = EXT / "off_alcohol/img" / p["code"]
        if not brand or not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG:
                yield {"path": f, "source": "off", "brand": brand, "product": p["product_name"],
                       "product_key": f"off:{p['code']}", "size": p["quantity"], "group": p.get("group", ""), "gallery": 1}
    # Retailer catalogues: product = retailer id, front packshot
    for src in ("saq", "systembolaget", "drinkfinder"):
        for p in csv.DictReader(open(EXT / f"retail_{src}/products.csv")):
            d = EXT / f"retail_{src}/img" / p["product_id"]
            if not p["brand"] or not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMG:
                    yield {"path": f, "source": src, "brand": p["brand"], "product": p["name"],
                           "product_key": f"{src}:{p['product_id']}", "size": p["volume_ml"] and f"{p['volume_ml']} ml",
                           "group": p["category"], "gallery": 1}
    # In-the-wild photos: brand only (weak label), training only, never gallery
    for p in csv.DictReader(open(EXT / "wild/manifest.csv")):
        f = EXT / "wild" / p["file"]
        if f.exists():
            yield {"path": f, "source": "wild", "brand": p["brand"], "product": "", "product_key": "",
                   "size": "", "group": "", "gallery": 0}
    # Russian beer bottles/cans: class = brand + variant
    for split in ("train", "validation", "test"):
        for d in sorted((EXT / "beer-classification-ru/images" / split).iterdir()):
            for f in sorted(d.iterdir()):
                if f.suffix.lower() in IMG:
                    yield {"path": f, "source": "beer_ru", "brand": d.name.replace("_", " "), "product": d.name,
                           "product_key": f"beer_ru:{d.name}", "size": "", "group": "beer", "gallery": 1}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = list(rows())
    for r in out:
        r["path"] = str(r["path"])
        r["brand_key"] = key(r["brand"])
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "brand", "brand_key", "product", "product_key", "size", "group", "gallery"])
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(f"{len(out)} images, {len({r['brand_key'] for r in out})} brands, "
          f"{len({r['product_key'] for r in out if r['product_key']})} products")
    print(Counter(r["source"] for r in out))
    per_brand = Counter(r["brand_key"] for r in out)
    print("brands with 100+ images:", sum(v >= 100 for v in per_brand.values()),
          " 30+:", sum(v >= 30 for v in per_brand.values()))


if __name__ == "__main__":
    main()
