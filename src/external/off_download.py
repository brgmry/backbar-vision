"""Download OFF images listed in data/external/off_alcohol/images.csv.

Resume-safe: skips files already on disk. UK products first. Non-image responses skipped.
Usage: off_download.py [workers]
"""
import csv, sys, time, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2] / "data/external/off_alcohol"
UA = "bottle-vision-hackathon/0.1 (you@example.com)"
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
local = threading.local()
stats = {"ok": 0, "skip": 0, "bad": 0, "err": 0, "bytes": 0}
lock = threading.Lock()


def session():
    if not hasattr(local, "s"):
        local.s = requests.Session()
        local.s.headers["User-Agent"] = UA
    return local.s


def fetch(row):
    code, name, url = row
    out = ROOT / "img" / code / name
    if out.exists():
        with lock: stats["skip"] += 1
        return
    for attempt in range(4):
        try:
            r = session().get(url, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt * 2); continue
            ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") \
                and r.content[:3] == b"\xff\xd8\xff"
            if not ok:
                with lock: stats["bad"] += 1
                return
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part")
            tmp.write_bytes(r.content); tmp.rename(out)
            with lock:
                stats["ok"] += 1; stats["bytes"] += len(r.content)
            return
        except requests.RequestException:
            time.sleep(2 ** attempt)
    with lock: stats["err"] += 1


def main():
    uk = {r["code"] for r in csv.DictReader(open(ROOT / "products.csv")) if r["uk"] == "1"}
    rows = [(r["code"], r["name"], r["url"]) for r in csv.DictReader(open(ROOT / "images.csv"))]
    rows.sort(key=lambda r: (r[0] not in uk, not r[1].startswith("front_")))
    total, t0 = len(rows), time.time()
    print(f"{total} images, {WORKERS} workers", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, _ in enumerate(ex.map(fetch, rows), 1):
            if i % 5000 == 0:
                el = time.time() - t0
                print(f"{i}/{total} {stats} {i/el:.0f}/s {stats['bytes']/1e9:.2f}GB", flush=True)
    print("done", stats, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
