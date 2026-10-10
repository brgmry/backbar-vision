"""Collect retail product catalogues (images + metadata) for pre-training.

Usage: python src/collect_retail.py <saq|systembolaget|drinkfinder> [--minutes N]

Whole categories only, never selected by data/labelled/. Resume-safe: products
whose image dir is complete are skipped, metadata pages are cached in meta.jsonl.
Polite: at most 1 request/second per host, identifying User-Agent.
"""
import csv, html, io, json, random, re, sys, threading, time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

UA = "bottle-vision-hackathon/0.1 (you@example.com)"
ROOT = Path(__file__).resolve().parent.parent / "data" / "external"
FIELDS = ["source", "product_id", "brand", "name", "category", "volume_ml", "abv", "url", "image_urls"]

S = requests.Session()
S.headers["User-Agent"] = UA
_last, _lock = {}, threading.Lock()


def get(url, **kw):
    host = urlparse(url).netloc
    with _lock:  # reserve the next 1.05s slot for this host, sleep outside the lock
        slot = max(time.time(), _last.get(host, 0) + 1.05)
        _last[host] = slot
    time.sleep(max(0, slot - time.time()))
    for attempt in range(3):
        try:
            r = S.get(url, timeout=60, **kw)
            if r.status_code in (429, 503):
                time.sleep(30 * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(5)
    return None


def vol_from(text):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(ml|cl|l)\b", text or "", re.I)
    if not m:
        return ""
    v, u = float(m.group(1)), m.group(2).lower()
    return int(round(v * {"ml": 1, "cl": 10, "l": 1000}[u]))


# ---------- SAQ (Magento storefront GraphQL, public) ----------
SAQ_Q = """{products(search:"",pageSize:%d,currentPage:%d){total_count items{__typename sku name
 media_gallery{url position disabled}
 custom_attributesV2{items{code ... on AttributeValue{value}}}}}}"""


def saq_page(page, size=100):
    r = get("https://www.saq.com/graphql", params={"query": SAQ_Q % (size, page)}, headers={"Store": "en"})
    if r is None or r.status_code != 200:
        print("saq meta fail", page, r and r.status_code, flush=True)
        return None, []
    data = (r.json().get("data") or {}).get("products") or {}
    rows = []
    for it in [i for i in data.get("items") or [] if i]:
        a = {x["code"]: x.get("value") for x in (it.get("custom_attributesV2") or {}).get("items", [])}
        desc = html.unescape(a.get("meta_description") or "")
        imgs = [m["url"] for m in sorted(it.get("media_gallery") or [], key=lambda m: m.get("position") or 0)
                if not m.get("disabled")]
        abv = a.get("pourcentage_alcool_par_volume") or ""
        rows.append(dict(source="saq", product_id=it["sku"], brand=html.unescape(a.get("nom_producteur") or ""),
                         name=it["name"], category=a.get("identite_produit") or desc.split(".")[0],
                         volume_ml=vol_from(desc.split("Size :")[-1]) if "Size :" in desc else "",
                         abv=f"{float(abv):g}" if abv else "", url=f"https://www.saq.com/en/{it['sku']}",
                         image_urls=imgs))
    return data.get("total_count"), rows


def saq_meta(out):
    """Generator: pages in shuffled order so a partial run spans the catalogue."""
    total, first = saq_page(1)
    pages = list(range(2, (total + 99) // 100 + 1))
    random.Random(0).shuffle(pages)
    yield from first
    for p in pages:
        yield from saq_page(p)[1]


# ---------- Systembolaget (product sitemaps + product page JSON) ----------
SB_CATS = ["ol", "vin", "sprit", "cider-blanddrycker", "alkoholfritt"]


def sb_meta(out):
    cache = out / "urls.json"
    if cache.exists():
        urls = json.loads(cache.read_text())
    else:
        urls = []
        for c in SB_CATS:
            r = get(f"https://www.systembolaget.se/sitemap-produkter-{c}.xml")
            urls += re.findall(r"<loc>([^<]+)</loc>", r.text)
        cache.write_text(json.dumps(urls))
    rows = []
    for u in urls:
        m = re.search(r"/produkt/([^/]+)/(.+)-(\d+)/?$", u)
        if m:
            rows.append(dict(source="systembolaget", product_id=m.group(3), category=m.group(1), url=u,
                             _slug=m.group(2)))
    return rows


def sb_fill(d):
    r = get(d["url"])
    if r is None or r.status_code != 200:
        return False
    h = r.text
    pid = re.search(r'"productId":"(\d+)"', h)
    if not pid:
        return False

    def f(k, pat=r'"%s":"((?:[^"\\]|\\.)*)"'):
        m = re.search(pat % k, h)
        return json.loads(f'"{m.group(1)}"') if m else ""

    bold, thin = f("productNameBold"), f("productNameThin")
    d.update(brand=f("producerName"), name=" ".join(x for x in (bold, thin) if x) or d["_slug"],
             category=f("categoryLevel2") or d["category"],
             volume_ml=(re.search(r'"volume":([0-9.]+)', h) or [None, ""])[1],
             abv=(re.search(r'"alcoholPercentage":([0-9.]+)', h) or [None, ""])[1],
             image_urls=[f"https://product-cdn.systembolaget.se/productimages/{pid.group(1)}/{pid.group(1)}_800.png"])
    return True


# ---------- Drinkfinder (Shopify products.json) ----------
def df_meta(out):
    cache = out / "meta.jsonl"
    if cache.exists():
        return [json.loads(l) for l in cache.open()]
    rows, page = [], 1
    with cache.open("w") as f:
        while True:
            r = get(f"https://www.drinkfinder.co.uk/products.json?limit=250&page={page}")
            ps = r.json().get("products", []) if r is not None and r.status_code == 200 else []
            if not ps:
                break
            for p in ps:
                body = re.sub("<[^>]+>", " ", p.get("body_html") or "")
                abv = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:abv|vol)", body + " " + p["title"], re.I)
                d = dict(source="drinkfinder", product_id=str(p["id"]), brand=p.get("vendor") or "",
                         name=p["title"], category=p.get("product_type") or ",".join(p.get("tags") or []),
                         volume_ml=vol_from(p["title"]), abv=abv.group(1) if abv else "",
                         url=f"https://www.drinkfinder.co.uk/products/{p['handle']}",
                         image_urls=[i["src"].split("?")[0] + "?width=1000" for i in p.get("images") or []])
                rows.append(d)
                f.write(json.dumps(d) + "\n")
            page += 1
    return rows


def save_img(content, path):
    im = Image.open(io.BytesIO(content))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    im.convert("RGB").save(path, "JPEG", quality=90)


def main():
    src = sys.argv[1]
    minutes = float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else 1e9
    deadline = time.time() + minutes * 60
    out = ROOT / f"retail_{src}"
    (out / "img").mkdir(parents=True, exist_ok=True)
    rows = {"saq": saq_meta, "systembolaget": sb_meta, "drinkfinder": df_meta}[src](out)
    if isinstance(rows, list):
        random.Random(0).shuffle(rows)  # spread partial runs across categories
    csv_path = out / "products.csv"
    done = set()
    if csv_path.exists():
        done = {r["product_id"] for r in csv.DictReader(csv_path.open())}
    new = not csv_path.exists()
    fh = csv_path.open("a", newline="")
    w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
    if new:
        w.writeheader()
    print(src, "products", len(rows) if isinstance(rows, list) else "streamed", "already done", len(done), flush=True)
    wlock, it = threading.Lock(), iter(rows)

    def one(d):
        if src == "systembolaget" and not sb_fill(d):
            return
        got = []
        pdir = out / "img" / d["product_id"]
        for i, u in enumerate(d.get("image_urls") or []):
            r = get(u)
            if r is None or r.status_code != 200:
                continue
            try:
                pdir.mkdir(exist_ok=True)
                save_img(r.content, pdir / f"{i}.jpg")
                got.append(u)
            except Exception as e:
                print("bad image", u, e, flush=True)
        if not got:
            return
        d["image_urls"] = " ".join(got)
        with wlock:
            w.writerow(d)
            fh.flush()
            done.add(d["product_id"])
            if len(done) % 200 == 0:
                print(src, "saved", len(done), time.strftime("%H:%M:%S"), flush=True)

    def worker():
        while time.time() < deadline:
            with wlock:
                d = next(it, None)
                while d is not None and d["product_id"] in done:
                    d = next(it, None)
            if d is None:
                return
            try:
                one(d)
            except Exception as e:
                print("error", d.get("product_id"), e, flush=True)

    ts = [threading.Thread(target=worker) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print(src, "finished, total saved", len(done), flush=True)


if __name__ == "__main__":
    main()
