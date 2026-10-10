"""Collect in-the-wild bottle photos per brand for pre-training.

Brand list comes from data/external/brands.csv (built from trade rankings, never
from our ground truth). Sources: Wikimedia Commons, Openverse, DuckDuckGo/Bing images.
Output: data/external/wild/<slug>/<n>.jpg + data/external/wild/manifest.csv.

usage: python src/collect_wild.py [max_brands] [per_brand]
"""
import csv, io, sys, threading, time, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash, requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "external"
OUT = ROOT / "wild"
UA = "bottle-vision-hackathon/0.1 (you@example.com)"
MIN_SIDE, MAX_SIDE = 200, 1024

S = requests.Session()
S.headers["User-Agent"] = UA

class Throttle:
    def __init__(self, gap):
        self.gap, self.lock, self.t = gap, threading.Lock(), 0.0
    def wait(self):
        with self.lock:
            d = self.t + self.gap - time.time()
            if d > 0: time.sleep(d)
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
    except Exception:
        return []
    out = []
    for p in (r.get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        lic = (ii.get("extmetadata", {}).get("LicenseShortName", {}) or {}).get("value", "")
        if ii.get("thumburl"):
            out.append(("commons", ii["thumburl"], lic, ii.get("descriptionurl", "")))
    return out


def openverse(brand, n=20):
    with GH_LOCK:
        if OV_BUDGET[0] <= 0: return []
        OV_BUDGET[0] -= 1
    OV_T.wait()
    try:
        r = S.get("https://api.openverse.org/v1/images/", timeout=15,
                  params={"q": f"{brand} bottle", "page_size": n})
        if r.status_code != 200:
            if r.status_code == 429: OV_BUDGET[0] = 0
            return []
        res = r.json().get("results", [])
    except Exception:
        return []
    return [("openverse", x["url"], f'{x.get("license","")} {x.get("license_version","")}'.strip(),
             x.get("foreign_landing_url", "")) for x in res if x.get("url")]


def ddg(brand, query_tpls, n=40):
    from ddgs import DDGS
    out = []
    for q in query_tpls:
        DDG_T.wait()
        try:
            res = DDGS().images(q.format(b=brand), max_results=n)
        except Exception:
            time.sleep(5)
            continue
        out += [("ddg", x["image"], "unknown", x.get("url", ""), x.get("thumbnail")) for x in res]
    return out


def fetch(c):
    urls = [c[1]] + ([c[4]] if len(c) > 4 and c[4] else [])
    for u in urls:
        try:
            r = S.get(u, timeout=6, stream=True)
            if r.status_code != 200: continue
            b = r.raw.read(15_000_000, decode_content=True)
            im = Image.open(io.BytesIO(b)); im.load()
            im = im.convert("RGB")
            if min(im.size) < MIN_SIDE: continue
            im.thumbnail((MAX_SIDE, MAX_SIDE))
            return im, u
        except Exception:
            continue
    return None, None


def do_brand(row, per_brand):
    slug, brand = row["slug"], row["brand"]
    d = OUT / slug
    if (d / "_done").exists(): return
    d.mkdir(parents=True, exist_ok=True)
    cands = commons(brand) + openverse(brand)
    cands += ddg(brand, ["{b} bottle bar", "{b} bottle cocktail", "{b} bottle hand"])
    seen_url, uniq = set(), []
    for c in cands:
        if c[1] not in seen_url: seen_url.add(c[1]); uniq.append(c)
    hashes, rows, k = [], [], 0
    with ThreadPoolExecutor(8) as ex:
        for c, (im, used) in zip(uniq, ex.map(fetch, uniq)):
            if k >= per_brand: break
            if im is None: continue
            h = imagehash.phash(im)
            if any(h - x <= 6 for x in hashes): continue
            with GH_LOCK:
                if any(h - x <= 2 for x in GLOBAL_HASHES): continue
                GLOBAL_HASHES.append(h)
            hashes.append(h)
            fn = d / f"{k}.jpg"
            im.save(fn, quality=88)
            rows.append({"brand": brand, "slug": slug, "file": f"{slug}/{k}.jpg", "source": c[0],
                         "url": used, "page": c[3], "licence": c[2], "w": im.size[0], "h": im.size[1],
                         "phash": str(h)})
            k += 1
    with open(d / "_manifest.json", "w") as f: json.dump(rows, f)
    (d / "_done").touch()
    print(f"{slug}: {k} (cands {len(uniq)})", flush=True)


def main():
    max_brands = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    per_brand = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    brands = list(csv.DictReader(open(ROOT / "brands.csv")))
    brands.sort(key=lambda r: -len(r["sources"].split(";")))  # stable: most-cited first
    brands = brands[:max_brands]
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(lambda r: do_brand(r, per_brand), brands))
    merge()


def merge():
    rows = []
    for m in sorted(OUT.glob("*/_manifest.json")): rows += json.load(open(m))
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["brand", "slug", "file", "source", "url", "page", "licence", "w", "h", "phash"])
        w.writeheader(); w.writerows(rows)
    print("total", len(rows))


if __name__ == "__main__":
    if sys.argv[1:2] == ["merge"]: merge()
    else: main()
