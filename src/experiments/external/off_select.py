"""Select bar-relevant products (alcohol + mixers) from the Open Food Facts parquet dump.

Selection is by OFF category tags only, never by our inventory.
Input:  data/external/off_alcohol/raw/food.parquet (HF openfoodfacts/product-database)
Output: data/external/off_alcohol/products.csv, one row per product,
        plus images.csv, one row per image to fetch (code, kind, name, url).
Usage: python src/experiments/external/off_select.py [max_other]   (cap on non-front images per product, default no cap)
"""
import argparse
import csv
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3] / "data/external/off_alcohol"
SRC = ROOT / "raw/food.parquet"
BASE = "https://images.openfoodfacts.org/images/products"

# Exact category tags that make a product bar-relevant.
ALCOHOL = ["en:alcoholic-beverages", "en:distilled-beverages", "en:artisanal-spirits", "en:liqueurs",
           "en:wines", "en:sparkling-wines", "en:fortified-wines", "en:champagnes", "en:prosecco",
           "en:beers", "en:lagers", "en:stouts", "en:ciders", "en:hard-seltzer", "en:hard-seltzers",
           "en:whisky", "en:vodka", "en:gins", "en:rums", "en:tequilas", "en:cognac", "en:brandys",
           "en:vermouth", "en:aperitif", "en:sherry"]
# Mixers: tonic, soda, colas, ginger beer, soda water, bitters, cocktail mixers, alcohol-free versions.
# Plain still waters and squash syrups are left out.
MIXERS = ["en:tonic-water", "en:tonic-drink-with-sugar", "en:tonic-drink-with-sugar-and-artificial-sweeteners",
          "en:sodas", "en:colas", "en:lemonades", "en:carbonated-waters", "en:ginger-beer", "en:ginger-ales",
          "en:bitters", "en:cocktail-mixers", "en:mixer", "en:alcohol-free-cocktails",
          "en:non-alcoholic-beers", "en:non-alcoholic-wines"]


def split_code(code: str) -> str:
    # OFF scheme: pad to 13 digits, then 3/3/3/rest. Codes of 8 digits or fewer are not split.
    if len(code) <= 8:
        return code
    c = code.zfill(13)
    return "/".join([c[0:3], c[3:6], c[6:9], c[9:]])


def main(max_other: int):
    con = duckdb.connect()
    tags = ALCOHOL + MIXERS
    tag_sql = "[" + ",".join(f"'{t}'" for t in tags) + "]"
    rows = con.sql(f"""
        select code, brands, product_name, quantity, categories, categories_tags,
               countries_tags, images
        from read_parquet('{SRC}')
        where categories_tags is not null
          and len(list_intersect(categories_tags, {tag_sql})) > 0
          and images is not null and len(images) > 0
    """).fetchall()
    with open(ROOT / "products.csv", "w", newline="") as prod_f, open(ROOT / "images.csv", "w", newline="") as img_f:
        n_img = write(rows, csv.writer(prod_f), csv.writer(img_f), max_other)
    print(f"products {len(rows)}  images {n_img}")


def write(rows, pw, iw, max_other: int) -> int:
    """One products.csv row per product, one images.csv row per image to fetch. Returns the image count."""
    alc = set(ALCOHOL)
    pw.writerow(["code", "brands", "product_name", "quantity", "categories", "countries",
                 "group", "uk", "n_images", "image_ids"])
    iw.writerow(["code", "name", "url"])
    n_img = 0
    for code, brands, pname, qty, cats, ctags, countries, images in rows:
        if not code or not re.fullmatch(r"\d+", code):
            continue
        name = ""
        if pname:
            byl = {d["lang"]: d["text"] for d in pname if d.get("text")}
            name = byl.get("main") or byl.get("en") or next(iter(byl.values()), "")
        group = "alcohol" if alc & set(ctags) else "mixer"
        path = split_code(code)
        fronts, raws = [], []
        for im in images:
            key = im["key"]
            if key.isdigit():
                raws.append(int(key))
            elif key.startswith("front_") and im.get("rev") is not None:
                fronts.append((key, im["rev"]))
        # front_en first, then main-language fronts; keep all fronts (different languages differ)
        fronts.sort(key=lambda kr: (kr[0] != "front_en", kr[0]))
        raws.sort(reverse=True)  # newest uploads first
        ids = [f"{k}.{r}" for k, r in fronts] + [str(r) for r in raws[:max_other]]
        for k, r in fronts:
            iw.writerow([code, f"{k}.jpg", f"{BASE}/{path}/{k}.{r}.400.jpg"])
        for r in raws[:max_other]:
            iw.writerow([code, f"{r}.jpg", f"{BASE}/{path}/{r}.400.jpg"])
        n_img += len(ids)
        uk = int(bool(countries) and "en:united-kingdom" in countries)
        pw.writerow([code, brands or "", name, qty or "", cats or "", ",".join(countries or []),
                     group, uk, len(ids), " ".join(ids)])
    return n_img


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("max_other", nargs="?", type=int, default=10**6)
    main(ap.parse_args().max_other)
