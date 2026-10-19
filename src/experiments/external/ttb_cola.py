"""Polite scraper for TTB COLA public registry spirits label images.
Step 1: list COLAs via basic search + CSV export, weekly windows, spirits class ranges.
Step 2: per COLA, open the printable form and download every label attachment.

Usage: python src/experiments/external/ttb_cola.py [target] [--out data/external/ttb-cola-spirits]   (rerun to resume)
"""
import argparse
import csv
import io
import random
import re
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import partial
from pathlib import Path

from curlsess import CurlSession

ROOT = Path(__file__).resolve().parents[3]
BASE = "https://ttbonline.gov/colasonline/"
UA = "Mozilla/5.0 (academic dataset research; low rate)"
IMG = re.compile(r'src="/colasonline/(publicViewAttachment\.do\?filename=([^"&]*)&filetype=l)"')
EXT = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/tiff": "tif"}
WORKERS = 8
local = threading.local()
lock = threading.Lock()


def session():
    s = CurlSession(UA)
    s.get(BASE + "publicSearchColasBasic.do", timeout=60)
    return s


def search(s, d0, d1, c0, c1):
    data = {"searchCriteria.dateCompletedFrom": d0.strftime("%m/%d/%Y"),
            "searchCriteria.dateCompletedTo": d1.strftime("%m/%d/%Y"),
            "searchCriteria.productOrFancifulName": "", "searchCriteria.productNameSearchType": "E",
            "searchCriteria.classTypeFrom": c0, "searchCriteria.classTypeTo": c1, "searchCriteria.originCode": ""}
    s.post(BASE + "publicSearchColasBasicProcess.do?action=search", data=data, timeout=120)
    time.sleep(1)
    r = s.get(BASE + "publicSaveSearchResultsToFile.do?path=/publicSearchColasBasicProcess", timeout=120)
    rows = list(csv.reader(io.StringIO(r.text)))
    out = []
    for row in rows[1:]:
        if len(row) >= 10:
            row[0] = row[0].strip("'")
            out.append(row)
    time.sleep(1)
    return rows[0] if rows else None, out


def list_colas(index_path: Path, target: int) -> None:
    """Weekly windows every 4 weeks from 2024-01, spirits classes 100-699 plus agave 977-983; a random sample of
    `target` of the COLAs found goes to index.csv."""
    s, allrows, header = session(), {}, None
    random.seed(0)
    starts = [date(2024, 1, 1) + timedelta(weeks=w) for w in range(0, 140, 4)]
    for st in starts:
        for c0, c1 in (("100", "699"), ("977", "983")):
            h, rows = search(s, st, st + timedelta(days=6), c0, c1)
            header = header or h
            for r in rows:
                allrows[r[0]] = r
            print(st, c0, len(rows), "total", len(allrows), flush=True)
        if len(allrows) >= target * 1.3:
            break
    ids = sorted(allrows)
    random.shuffle(ids)
    ids = ids[:target]
    with open(index_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header[:10])
        for i in sorted(ids):
            w.writerow(allrows[i][:10])


def work(tid, out: Path, meta_path: Path):
    if not hasattr(local, "s"):
        local.s = session()
    s = local.s
    try:
        r = s.get(BASE + f"viewColaDetails.do?action=publicFormDisplay&ttbid={tid}", timeout=60)
        time.sleep(0.5)
        rows = []
        for k, (path, name) in enumerate(IMG.findall(r.text)):
            url = BASE + path.replace(" ", "%20")
            im = s.get(url, timeout=120)
            time.sleep(0.5)
            if im.status_code != 200 or not im.headers.get("content-type", "").startswith("image"):
                continue
            fn = f"images/{tid}_{k}.{EXT.get(im.headers['content-type'].split(';')[0], 'bin')}"
            (out / fn).write_bytes(im.content)
            rows.append((tid, fn, urllib.parse.unquote(name).replace(",", " ")))
        if not rows:
            rows.append((tid, "", "NO_IMAGES"))  # recorded so a resumed run does not ask again
        with lock, open(meta_path, "a") as f:
            for row in rows:
                f.write(",".join(row) + "\n")
    except (OSError, ValueError) as e:  # curl failed or answered garbage: log it, the next run retries
        print("ERR", tid, e, flush=True)
        time.sleep(5)


def main(out: Path, target: int) -> None:
    (out / "images").mkdir(parents=True, exist_ok=True)
    index_path, meta_path = out / "index.csv", out / "images.csv"
    if not index_path.exists():
        list_colas(index_path, target)
    with open(index_path, newline="") as f:
        ids = [r["TTB ID"] for r in csv.DictReader(f)]
    done = set()
    if meta_path.exists():
        with open(meta_path, newline="") as f:
            done = {r["ttb_id"] for r in csv.DictReader(f)}
    else:
        meta_path.write_text("ttb_id,file,orig_name\n")
    todo = [i for i in ids if i not in done]
    print("todo", len(todo), flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        for n, _ in enumerate(ex.map(partial(work, out=out, meta_path=meta_path), todo)):
            if n % 100 == 0:
                print("done", n, flush=True)
    print("FINISHED", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", nargs="?", type=int, default=3000, help="COLAs to sample")
    ap.add_argument("--out", type=Path, default=ROOT / "data/external/ttb-cola-spirits")
    args = ap.parse_args()
    main(args.out, args.target)
