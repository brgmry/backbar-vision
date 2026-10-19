"""Download OFF images listed in data/external/off_alcohol/images.csv.

Resume-safe: skips files already on disk. UK products first. Non-image responses skipped.
Needs CONTACT_EMAIL in .env (it goes in the User-Agent).
Usage: python src/experiments/external/off_download.py [workers]
"""
import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/experiments/, for contact.py
from contact import user_agent  # noqa: E402

ROOT = Path(__file__).resolve().parents[3] / "data/external/off_alcohol"
JPEG_MAGIC = b"\xff\xd8\xff"
local = threading.local()
stats = {"ok": 0, "skip": 0, "bad": 0, "err": 0, "bytes": 0}
lock = threading.Lock()


def session(ua: str):
    if not hasattr(local, "s"):
        local.s = requests.Session()
        local.s.headers["User-Agent"] = ua
    return local.s


def count(key, n=1):
    with lock:
        stats[key] += n


def fetch(row, ua: str):
    code, name, url = row
    out = ROOT / "img" / code / name
    if out.exists():
        count("skip")
        return
    for attempt in range(4):
        try:
            r = session(ua).get(url, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt * 2)
                continue
            ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("image/") \
                and r.content[:3] == JPEG_MAGIC
            if not ok:
                count("bad")
                return
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(out)
            count("ok")
            count("bytes", len(r.content))
            return
        except requests.RequestException:
            time.sleep(2 ** attempt)
    count("err")


def main(workers: int):
    ua = user_agent()
    with open(ROOT / "products.csv", newline="") as f:
        uk = {r["code"] for r in csv.DictReader(f) if r["uk"] == "1"}
    with open(ROOT / "images.csv", newline="") as f:
        rows = [(r["code"], r["name"], r["url"]) for r in csv.DictReader(f)]
    rows.sort(key=lambda r: (r[0] not in uk, not r[1].startswith("front_")))
    total, t0 = len(rows), time.time()
    print(f"{total} images, {workers} workers", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(partial(fetch, ua=ua), rows), 1):
            if i % 5000 == 0:
                el = time.time() - t0
                print(f"{i}/{total} {stats} {i / el:.0f}/s {stats['bytes'] / 1e9:.2f}GB", flush=True)
    print("done", stats, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workers", nargs="?", type=int, default=12)
    main(ap.parse_args().workers)
