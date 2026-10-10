"""Download a broad UK product catalogue from Open Food Facts: what a restaurant or bar keeps in stock.

Categories are chosen by venue type, never by looking at our answers. ODbL data, CC BY-SA images.
Usage: python src/catalogue.py   -> data/catalogue/products.csv + data/catalogue/img/<code>.jpg
"""
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

UA = {"User-Agent": "bottle-vision-hackathon/0.1 (you@example.com)"}
CATEGORIES = ["beverages", "alcoholic-beverages", "sauces", "condiments", "milks", "syrups"]
FIELDS = "code,product_name,brands,quantity,categories_tags,image_front_url"
OUT = Path("data/catalogue")


def search(category: str) -> list[dict]:
    products, page = [], 1
    while True:
        r = requests.get("https://world.openfoodfacts.org/api/v2/search", headers=UA, timeout=60, params={
            "countries_tags_en": "united-kingdom", "categories_tags_en": category,
            "fields": FIELDS, "page_size": 100, "page": page})
        if r.status_code != 200:
            time.sleep(30)
            continue
        batch = r.json().get("products", [])
        products += batch
        print(f"{category} page {page}: {len(batch)}", flush=True)
        if len(batch) < 100:
            return products
        page += 1
        time.sleep(6.5)  # search API allows ~10 requests a minute, 100 per page max


def download(p: dict) -> bool:
    path = OUT / "img" / f"{p['code']}.jpg"
    if path.exists():
        return True
    try:
        r = requests.get(p["image_front_url"], headers=UA, timeout=30)
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            path.write_bytes(r.content)
            return True
    except requests.RequestException:
        pass
    return False


def main() -> None:
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    seen = {}
    for c in CATEGORIES:
        for p in search(c):
            if p.get("image_front_url") and p.get("product_name") and p["code"] not in seen:
                seen[p["code"]] = p
        time.sleep(7)
    products = list(seen.values())
    print(f"{len(products)} products with a front image, downloading", flush=True)
    with ThreadPoolExecutor(8) as pool:
        ok = list(pool.map(download, products))
    with (OUT / "products.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "brands", "product_name", "quantity", "categories"])
        w.writeheader()
        for p, good in zip(products, ok):
            if good:
                w.writerow({"code": p["code"], "brands": p.get("brands", ""), "product_name": p["product_name"],
                            "quantity": p.get("quantity", ""), "categories": ",".join(p.get("categories_tags", [])[-3:])})
    print(f"{sum(ok)} images saved", flush=True)


if __name__ == "__main__":
    main()
