"""Polite scraper for TTB COLA public registry spirits label images.
Step 1: list COLAs via basic search + CSV export, weekly windows, spirits class ranges.
Step 2: per COLA, open the printable form and download every label attachment.
"""
import csv, io, os, random, re, sys, time, threading, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from curlsess import CurlSession

OUT = "~/dev/bottle-vision/data/external/ttb-cola-spirits"
BASE = "https://ttbonline.gov/colasonline/"
UA = "Mozilla/5.0 (academic dataset research; low rate)"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
os.makedirs(f"{OUT}/images", exist_ok=True)

def session():
    s = CurlSession(UA)
    s.get(BASE + "publicSearchColasBasic.do", timeout=60); return s

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
            row[0] = row[0].strip("'"); out.append(row)
    time.sleep(1)
    return rows[0] if rows else None, out

index_path = f"{OUT}/index.csv"
if not os.path.exists(index_path):
    s = session(); allrows = {}; header = None
    # weekly windows spread over 2024-01 .. 2026-09; spirits classes 100-699 plus agave 977-983
    random.seed(0)
    starts = [date(2024, 1, 1) + timedelta(weeks=w) for w in range(0, 140, 4)]
    for st in starts:
        for c0, c1 in (("100", "699"), ("977", "983")):
            h, rows = search(s, st, st + timedelta(days=6), c0, c1)
            header = header or h
            for r in rows: allrows[r[0]] = r
            print(st, c0, len(rows), "total", len(allrows), flush=True)
        if len(allrows) >= TARGET * 1.3: break
    ids = sorted(allrows); random.shuffle(ids); ids = ids[:TARGET]
    with open(index_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header[:10]); [w.writerow(allrows[i][:10]) for i in sorted(ids)]

ids = [r["TTB ID"] for r in csv.DictReader(open(index_path))]
lock = threading.Lock(); meta_path = f"{OUT}/images.csv"
done = set()
if os.path.exists(meta_path):
    done = {r["ttb_id"] for r in csv.DictReader(open(meta_path))}
else:
    open(meta_path, "w").write("ttb_id,file,orig_name\n")
local = threading.local()
IMG = re.compile(r'src="/colasonline/(publicViewAttachment\.do\?filename=([^"&]*)&filetype=l)"')

def work(tid):
    if not hasattr(local, "s"): local.s = session()
    s = local.s
    try:
        r = s.get(BASE + f"viewColaDetails.do?action=publicFormDisplay&ttbid={tid}", timeout=60)
        time.sleep(0.5)
        rows = []
        for k, (path, name) in enumerate(IMG.findall(r.text)):
            url = BASE + path.replace(" ", "%20")
            im = s.get(url, timeout=120)
            time.sleep(0.5)
            if im.status_code != 200 or not im.headers.get("content-type", "").startswith("image"): continue
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/tiff": "tif"}.get(im.headers["content-type"].split(";")[0], "bin")
            fn = f"images/{tid}_{k}.{ext}"
            open(f"{OUT}/{fn}", "wb").write(im.content)
            rows.append((tid, fn, urllib.parse.unquote(name).replace(",", " ")))
        if not rows: rows.append((tid, "", "NO_IMAGES"))
        with lock:
            with open(meta_path, "a") as f:
                for row in rows: f.write(",".join(row) + "\n")
    except Exception as e:
        print("ERR", tid, e, flush=True); time.sleep(5)

todo = [i for i in ids if i not in done]
print("todo", len(todo), flush=True)
with ThreadPoolExecutor(8) as ex:
    for n, _ in enumerate(ex.map(work, todo)):
        if n % 100 == 0: print("done", n, flush=True)
print("FINISHED", flush=True)
