"""Build the training index from external data only: one row per image with brand and product keys.

Sources (source tag): off, saq/systembolaget/drinkfinder (retailers), wild + wild_verified, beer_ru, whiskey,
stperez, beerbb (box crops), rp2k (alcohol classes), ttb (label artwork). Not indexed: openimages-bottles and
imagenet-bottles (no brand labels), wines-label-dataset (label boxes only, no wine identity), winesensed and
wine-images-126k (images live only as bytes inside parquet, nothing on disk to point a path at).

Columns: group is the coarse bar category (spirit, liqueur, wine, beer, mixer, other); category is the source's own
finer category text. gallery 1 = usable as a reference image of a known product, 0 = training only.
Every image within pHash distance holdout.MAX_PHASH_DIST of a bar test photo is left out, whatever its source;
the ones left out are listed in data/train/test_overlap.csv.
Usage: python src/train_index.py   -> data/train/index.csv (writes beerbb crops on first run)
"""
import csv
import json
import re
import unicodedata
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from holdout import MAX_PHASH_DIST, image_phash, test_distance, test_hashes

EXT = Path("data/external")
OUT = Path("data/train")
IMG = {".jpg", ".jpeg", ".png", ".webp", ".gif"}  # what the external sources ship; TTB labels include GIFs
GROUPS = ("spirit", "liqueur", "wine", "beer", "mixer", "other")
FIELDS = ["path", "source", "brand", "brand_key", "product", "product_key", "size", "group", "category", "gallery"]

# Coarse group from category text (English, French, Swedish, OFF tags). First match wins, so order matters.
GROUP_RULES = [
    ("mixer", r"non-alcoholic|alcohol-free|alkoholfri|tonic|\bsodas?\b|\bcolas?\b|lemonade|ginger-?(ale|beer)|"
              r"carbonated|\bmixers?\b|syrups?\b|\bbitters\b|\bwaters?\b"),
    ("liqueur", r"liqueur|lik.r|cordial|cream beverage|vermouth|aperiti|\bbitter\b|amaro|punsch|schnapps|pastis|"
                r"gentian|limoncello"),
    ("spirit", r"distilled|spirit|whisk|vodka|\bgins?\b|genever|jenever|\brums?\b|\brom\b|tequila|mezcal|agave|"
               r"cognac|brand(y|ies|ys)|armagnac|calvados|grappa|pisco|absinth|akvavit|br.nnvin|sprit|eau de vie|"
               r"eau-de-vie|baijiu|cacha|\barak\b|foursquare|kilkerran"),
    ("wine", r"wine|\bvins?\b|vitt|r.tt|ros.|champagne|prosecco|\bports?\b|sherry|madeira|sake|starkvin|"
             r"mousserande|gl.gg|gl.hwein|\bmust\b|sauternes|muscat|pineau|maury|montilla|moscatel"),
    ("beer", r"beers?\b|\bales?\b|lager|stout|porter|pilsn|\b.l\b|veteol|vete.l|cider|cidre"),
]


def group_of(*texts: str, default: str = "other") -> str:
    """Try each text in turn (category first, product name as fallback)."""
    for text in texts:
        t = (text or "").lower().replace("en:artisanal-spirits", "")  # OFF tags craft beers with this one
        for g, pat in GROUP_RULES:
            if re.search(pat, t):
                return g
    return default


def key(s: str) -> str:
    """Normalised brand/product key: ascii, lowercase, alphanumerics only. Non-latin names keep their characters."""
    s = s or ""
    a = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    a = re.sub(r"[^a-z0-9]+", " ", a).strip()
    return a or re.sub(r"\s+", " ", s).strip().lower()


def first_brand(brands: str) -> str:
    return (brands or "").split(",")[0].strip()


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def images(d: Path):
    return [f for f in sorted(d.iterdir()) if f.is_file() and f.suffix.lower() in IMG] if d.is_dir() else []


def off_tags() -> dict:
    """OFF English category tags per barcode, read from the raw dump (products.csv only has localised text)."""
    raw = EXT / "off_alcohol/raw/food.parquet"
    if not raw.exists():
        return {}
    import duckdb
    con = duckdb.connect()
    codes = [(p["code"],) for p in read_rows(EXT / "off_alcohol/products.csv")]
    con.execute("create table c(code varchar)")
    con.executemany("insert into c values (?)", codes)
    return dict(con.sql(f"select p.code, array_to_string(p.categories_tags, ' ') from read_parquet('{raw}') p "
                        "join c using (code)").fetchall())


def off():
    tags = off_tags()
    for p in read_rows(EXT / "off_alcohol/products.csv"):
        brand = first_brand(p["brands"])
        if not brand:
            continue
        g = "mixer" if p.get("group") == "mixer" else \
            group_of(tags.get(p["code"], "") + " " + p["categories"], p["product_name"])
        for f in images(EXT / "off_alcohol/img" / p["code"]):
            yield {"path": f, "source": "off", "brand": brand, "product": p["product_name"],
                   "product_key": f"off:{p['code']}", "size": p["quantity"], "group": g,
                   "category": p["categories"], "gallery": 1}


def retail():
    for src in ("saq", "systembolaget", "drinkfinder"):
        for p in read_rows(EXT / f"retail_{src}/products.csv"):
            if not p["brand"]:
                continue
            for f in images(EXT / f"retail_{src}/img" / p["product_id"]):
                yield {"path": f, "source": src, "brand": p["brand"], "product": p["name"],
                       "product_key": f"{src}:{p['product_id']}", "size": p["volume_ml"] and f"{p['volume_ml']} ml",
                       "group": group_of(p["category"], p["name"]), "category": p["category"], "gallery": 1}


def wild():
    """Brand-only weak labels. Photos whose label text OCR confirmed (verified.csv) become wild_verified, gallery 1,
    with a brand-level product_key wild:<brand_key> (same convention as compare_backbones.py)."""
    verified = {r["file"] for r in read_rows(EXT / "wild/verified.csv")} \
        if (EXT / "wild/verified.csv").exists() else set()
    cats = {r["brand"]: r["category"] for r in read_rows(EXT / "brands.csv")}
    for p in read_rows(EXT / "wild/manifest.csv"):
        f = EXT / "wild" / p["file"]
        if f.is_file():
            v = p["file"] in verified
            yield {"path": f, "source": "wild_verified" if v else "wild", "brand": p["brand"], "product": "",
                   "product_key": f"wild:{key(p['brand'])}" if v else "", "size": "",
                   "group": group_of(cats.get(p["brand"], "").replace("_", " "), default="spirit"),
                   "category": cats.get(p["brand"], ""),
                   "gallery": int(v)}


def beer_ru():
    for split in ("train", "validation", "test"):
        for d in sorted((EXT / "beer-classification-ru/images" / split).iterdir()):
            for f in images(d):
                yield {"path": f, "source": "beer_ru", "brand": d.name.replace("_", " "), "product": d.name,
                       "product_key": f"beer_ru:{d.name}", "size": "", "group": "beer", "category": "beer",
                       "gallery": 1}


WHISKEY_BRANDS = {"glenallachie": "GlenAllachie", "glengrant": "Glen Grant", "glendronach": "GlenDronach",
                  "monkey": "Monkey Shoulder", "kione": "Kione Tiger", "balvenie": "The Balvenie"}


def whiskey():
    """Scotch whisky video frames; identity from the filename prefix via classes.csv."""
    root = EXT / "whiskey-classification"
    classes = {r["class_id"]: r["name"] for r in read_rows(root / "classes.csv")}
    for split in ("train", "val", "test"):
        for f in images(root / "dataset/images" / split):
            cid = f.name.split("_")[0]
            name = classes.get(cid)
            if not name:
                continue
            w = name.split()[0]
            brand = WHISKEY_BRANDS.get(w, w.title())
            yield {"path": f, "source": "whiskey", "brand": brand, "product": name.title(),
                   "product_key": f"whiskey:{cid}", "size": "", "group": "spirit", "category": "scotch whisky",
                   "gallery": 1}


STPEREZ_BRANDS = {"michelob_ultra": "Michelob Ultra", "corona": "Corona", "stella_artois": "Stella Artois",
                  "budweiser": "Budweiser", "modelo_especial": "Modelo Especial", "castle_lite": "Castle Lite",
                  "evian": "Evian", "Nestea": "Nestea", "Perrier": "Perrier", "Nestle_Pure_Life": "Nestle Pure Life",
                  "San_Pellegrino": "San Pellegrino", "powerade": "Powerade", "minutemaid": "Minute Maid",
                  "coca-cola": "Coca-Cola", "fanta": "Fanta", "sprite": "Sprite", "fuzetea": "Fuze Tea",
                  "Starbucks_Refreshers": "Starbucks Refreshers", "Starbucks_Doubleshot": "Starbucks Doubleshot",
                  "Starbucks_Bottled_Frappuccino": "Starbucks Frappuccino", "pureleaf": "Pure Leaf",
                  "pepsi": "Pepsi", "moutaindew": "Mountain Dew", "gatorade": "Gatorade", "tropicana": "Tropicana",
                  "liptonbrisk": "Lipton Brisk", "monsters": "Monster", "arizonaTeas": "Arizona"}


def stperez():
    """<Category>/<Company>/<brand>/{train,test}/; Balashi has no brand level. Beer is beer, the rest mixers."""
    root = EXT / "beverages-stperez"
    for split_dir in sorted(root.glob("*/*/**/")):
        if split_dir.name not in ("train", "test"):
            continue
        rel = split_dir.relative_to(root).parts
        cat, folder = rel[0], rel[-2]
        brand = "Balashi" if folder == "Balashi Brewery" else STPEREZ_BRANDS.get(folder, folder.replace("_", " "))
        for f in images(split_dir):
            yield {"path": f, "source": "stperez", "brand": brand, "product": brand,
                   "product_key": f"stperez:{key(brand)}", "size": "", "group": "beer" if cat == "Beer" else "mixer",
                   "category": cat, "gallery": 1}


def beerbb():
    """Crop every labelled box to beerbb-1k/crops/NNNN_k.jpg (written once) and index the crops."""
    from PIL import Image
    root = EXT / "beerbb-1k"
    crops = root / "crops"
    crops.mkdir(exist_ok=True)
    for js in sorted((root / "BeerBB-1K").glob("*.json")):
        with open(js) as f:
            boxes = json.load(f)
        im = None
        for k, b in enumerate(boxes):
            brand = (b.get("brand") or "").strip()
            if not brand or brand == "UNKNOWN" or b["w"] < 16 or b["h"] < 16:
                continue
            out = crops / f"{js.stem}_{k}.jpg"
            if not out.exists():
                if im is None:
                    with Image.open(js.with_suffix(".jpg")) as raw:
                        im = raw.convert("RGB")
                x0, y0 = max(0, b["x"]), max(0, b["y"])
                x1, y1 = min(im.width, b["x"] + b["w"]), min(im.height, b["y"] + b["h"])
                im.crop((x0, y0, x1, y1)).save(out, "JPEG", quality=95)
            yield {"path": out, "source": "beerbb", "brand": brand, "product": brand,
                   "product_key": f"beerbb:{key(brand)}", "size": "", "group": "beer", "category": "beer",
                   "gallery": 1}


RP2K_GROUP = {"beer": "beer", "baijiu": "spirit", "rice_wine": "wine", "other_alcohol": "other"}


def rp2k():
    """Alcohol SKUs only (alcohol_classes.csv). Brand = the Chinese class name. Cooking wine that slipped the
    keyword filter (烹饪, 厨用) is dropped."""
    root = EXT / "rp2k"
    for p in read_rows(root / "alcohol_classes.csv"):
        c = p["class"]
        if "烹饪" in c or "厨用" in c:
            continue
        for split in ("train", "test"):
            for f in images(root / "all" / split / c):
                yield {"path": f, "source": "rp2k", "brand": c, "product": c, "product_key": f"rp2k:{c}",
                       "size": "", "group": RP2K_GROUP.get(p["group"], "other"), "category": p["group"],
                       "gallery": 1}


def ttb():
    """US TTB COLA label artwork (front/back/neck), flat art not photos: training only."""
    root = EXT / "ttb-cola-spirits"
    meta = {r["TTB ID"]: r for r in read_rows(root / "index.csv")}
    for r in read_rows(root / "images.csv"):
        m, f = meta.get(r["ttb_id"]), root / r["file"]
        if not r["file"] or not m or not m["Brand Name"].strip() or not f.is_file():  # no file: COLA without labels
            continue
        desc = m["Class/Type Desc"]
        yield {"path": f, "source": "ttb", "brand": m["Brand Name"].strip().title(),
               "product": (m["Fanciful Name"].strip() or desc).title(), "product_key": f"ttb:{r['ttb_id']}",
               "size": "", "group": group_of(desc, default="spirit"), "category": desc, "gallery": 0}


SOURCES = [off, retail, wild, beer_ru, whiskey, stperez, beerbb, rp2k, ttb]


def without_test_overlap(rows: list[dict]) -> list[dict]:
    """The rows whose image is not a near copy of a bar test photo. The rest go to OUT/test_overlap.csv."""
    test = test_hashes()
    with Pool() as pool:
        hashes = pool.map(image_phash, [r["path"] for r in rows], chunksize=64)
    keep, dropped = [], []
    for r, h in zip(rows, hashes, strict=True):
        dist = test_distance(h, test) if h is not None else None  # an unreadable file is caught by embed_all
        if dist is not None and dist <= MAX_PHASH_DIST:
            dropped.append({"path": r["path"], "source": r["source"], "distance": dist})
        else:
            keep.append(r)
    with open(OUT / "test_overlap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source", "distance"])
        w.writeheader()
        w.writerows(dropped)
    per_source = ", ".join(f"{s} {n}" for s, n in Counter(d["source"] for d in dropped).most_common())
    print(f"left out {len(dropped)} images within pHash distance {MAX_PHASH_DIST} of a test photo"
          f"{': ' + per_source if per_source else ''}", flush=True)
    return keep


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for fn in SOURCES:
        n = len(out)
        out.extend(fn())
        print(f"{fn.__name__}: {len(out) - n} images", flush=True)
    for r in out:
        r["path"] = str(r["path"])
        r["brand_key"] = key(r["brand"])
    out = without_test_overlap(out)
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"\n{len(out)} images, {len({r['brand_key'] for r in out})} brands, "
          f"{len({r['product_key'] for r in out if r['product_key']})} products, "
          f"{sum(int(r['gallery']) for r in out)} gallery")
    print(f"{'source':15}{'images':>9}{'brands':>8}{'products':>10}{'gallery':>9}")
    for s, n in Counter(r["source"] for r in out).most_common():
        rs = [r for r in out if r["source"] == s]
        print(f"{s:15}{n:9}{len({r['brand_key'] for r in rs}):8}{len({r['product_key'] for r in rs if r['product_key']}):10}"
              f"{sum(int(r['gallery']) for r in rs):9}")
    g = Counter(r["group"] for r in out)
    print("per group:", ", ".join(f"{k} {g[k]}" for k in GROUPS))
    per_brand = Counter(r["brand_key"] for r in out)
    print("brands with 100+ images:", sum(v >= 100 for v in per_brand.values()),
          " 30+:", sum(v >= 30 for v in per_brand.values()))


if __name__ == "__main__":
    main()
