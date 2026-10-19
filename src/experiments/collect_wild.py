"""Collect in-the-wild bottle photos per brand for pre-training.

Brand list comes from data/external/brands.csv (built from trade rankings, never
from our ground truth). Sources: Wikimedia Commons, Openverse, DuckDuckGo/Bing images.
Output: data/external/wild/<slug>/<n>.jpg + data/external/wild/manifest.csv.
Needs CONTACT_EMAIL in .env (it goes in the User-Agent).

usage: python src/experiments/collect_wild.py [max_brands] [per_brand]
       python src/experiments/collect_wild.py merge   (rebuild manifest.csv from the per-brand manifests)
"""
import argparse
import csv
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import imagehash
import requests
import urllib3
from PIL import Image

from contact import user_agent

ROOT = Path(__file__).resolve().parents[2] / "data" / "external"
OUT = ROOT / "wild"
MIN_SIDE, MAX_SIDE = 200, 1024
MAX_BYTES = 15_000_000  # a bigger download is not a photo we want
SAME_BRAND_DIST = 6  # pHash distance at or under which two photos of one brand count as the same photo
ANY_BRAND_DIST = 2  # across brands only near-exact copies count: one shelf photo can rightly show two brands
FETCH_WORKERS, BRAND_WORKERS = 8, 12
FIELDS = ["brand", "slug", "file", "source", "url", "page", "licence", "w", "h", "phash"]
SEARCH_ERRORS = (requests.RequestException, ValueError)  # network, HTTP or a body that is not JSON
FETCH_ERRORS = (requests.RequestException, urllib3.exceptions.HTTPError, OSError, ValueError,
                Image.DecompressionBombError)

S = requests.Session()


class Candidate(NamedTuple):
    source: str
    url: str
    licence: str
    page: str
    thumb: str | None = None  # tried when the full image fails (DuckDuckGo only)


class Throttle:
    def __init__(self, gap):
        self.gap, self.lock, self.t = gap, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            d = self.t + self.gap - time.time()
            if d > 0:
                time.sleep(d)
            self.t = time.time()


OV_T, DDG_T, WM_T = Throttle(3.2), Throttle(1.0), Throttle(0.3)
OV_BUDGET = [180]  # anon limit is 200/day
GLOBAL_HASHES, GH_LOCK = [], threading.Lock()


def commons(brand, n=15):
    WM_T.wait()
    try:
        r = S.get("https://commons.wikimedia.org/w/api.php", timeout=15, params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f'"{brand}" bottle filetype:bitmap', "gsrnamespace": 6, "gsrlimit": n,
            "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": MAX_SIDE}).json()
    except SEARCH_ERRORS as e:
        print(f"commons search failed for {brand}: {e}", file=sys.stderr)
        return []
    out = []
    for p in (r.get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        lic = (ii.get("extmetadata", {}).get("LicenseShortName", {}) or {}).get("value", "")
        if ii.get("thumburl"):
            out.append(Candidate("commons", ii["thumburl"], lic, ii.get("descriptionurl", "")))
    return out


def openverse(brand, n=20):
    with GH_LOCK:
        if OV_BUDGET[0] <= 0:
            return []
        OV_BUDGET[0] -= 1
    OV_T.wait()
    try:
        r = S.get("https://api.openverse.org/v1/images/", timeout=15,
                  params={"q": f"{brand} bottle", "page_size": n})
        if r.status_code != 200:
            if r.status_code == 429:
                OV_BUDGET[0] = 0
            return []
        res = r.json().get("results", [])
    except SEARCH_ERRORS as e:
        print(f"openverse search failed for {brand}: {e}", file=sys.stderr)
        return []
    return [Candidate("openverse", x["url"], f'{x.get("license", "")} {x.get("license_version", "")}'.strip(),
                      x.get("foreign_landing_url", "")) for x in res if x.get("url")]


def ddg(brand, query_tpls, n=40):
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException
    out = []
    for q in query_tpls:
        DDG_T.wait()
        try:
            res = DDGS().images(q.format(b=brand), max_results=n)
        except DDGSException as e:
            print(f"ddg search failed for {brand}: {e}", file=sys.stderr)
            time.sleep(5)
            continue
        out += [Candidate("ddg", x["image"], "unknown", x.get("url", ""), x.get("thumbnail")) for x in res]
    return out


def fetch(c: Candidate):
    for u in [c.url] + ([c.thumb] if c.thumb else []):
        try:
            r = S.get(u, timeout=6, stream=True)
            if r.status_code != 200:
                continue
            im = Image.open(io.BytesIO(r.raw.read(MAX_BYTES, decode_content=True)))
            im.load()
            im = im.convert("RGB")
            if min(im.size) < MIN_SIDE:
                continue
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            return im, u
        except FETCH_ERRORS:
            continue
    return None, None


def do_brand(row, per_brand):
    slug, brand = row["slug"], row["brand"]
    d = OUT / slug
    if (d / "_done").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    cands = commons(brand) + openverse(brand)
    cands += ddg(brand, ["{b} bottle bar", "{b} bottle cocktail", "{b} bottle hand"])
    seen_url, uniq = set(), []
    for c in cands:
        if c.url not in seen_url:
            seen_url.add(c.url)
            uniq.append(c)
    hashes, rows = [], []
    with ThreadPoolExecutor(FETCH_WORKERS) as ex:
        for c, (im, used) in zip(uniq, ex.map(fetch, uniq), strict=True):
            if len(rows) >= per_brand:
                ex.shutdown(wait=False, cancel_futures=True)  # the rest are only queued: don't download them
                break
            if im is None:
                continue
            h = imagehash.phash(im)
            if any(h - x <= SAME_BRAND_DIST for x in hashes):
                continue
            with GH_LOCK:
                if any(h - x <= ANY_BRAND_DIST for x in GLOBAL_HASHES):
                    continue
                GLOBAL_HASHES.append(h)
            hashes.append(h)
            k = len(rows)
            im.save(d / f"{k}.jpg", quality=88)
            rows.append({"brand": brand, "slug": slug, "file": f"{slug}/{k}.jpg", "source": c.source,
                         "url": used, "page": c.page, "licence": c.licence, "w": im.size[0], "h": im.size[1],
                         "phash": str(h)})
    with open(d / "_manifest.json", "w") as f:
        json.dump(rows, f)
    (d / "_done").touch()
    print(f"{slug}: {len(rows)} (cands {len(uniq)})", flush=True)


def main(max_brands: int, per_brand: int) -> None:
    S.headers["User-Agent"] = user_agent()
    with open(ROOT / "brands.csv", newline="") as f:
        brands = list(csv.DictReader(f))
    brands.sort(key=lambda r: -len(r["sources"].split(";")))  # stable: most-cited first
    with ThreadPoolExecutor(BRAND_WORKERS) as ex:
        list(ex.map(lambda r: do_brand(r, per_brand), brands[:max_brands]))
    merge()


def merge():
    rows = []
    for m in sorted(OUT.glob("*/_manifest.json")):
        with open(m) as f:
            rows += json.load(f)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("total", len(rows))


if __name__ == "__main__":
    if sys.argv[1:2] == ["merge"]:
        merge()
    else:
        ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
        ap.add_argument("max_brands", nargs="?", type=int, default=300)
        ap.add_argument("per_brand", nargs="?", type=int, default=60)
        args = ap.parse_args()
        main(args.max_brands, args.per_brand)
