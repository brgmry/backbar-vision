"""Judge tab: blind A/B human judging of bottle IDs on the bar photos the two prediction files share, which have no
ground truth.

Mounted in api.py with app.include_router(judge_routes.router). Page at /judge, JSON and images under /api/judge.
Inputs are read again when they change (they may still be being written). Change the pair in CLAUDE / OURS below
or with the JUDGE_CLAUDE / JUDGE_OURS env vars. Claude: run 1. Ours: lowest run present.
Each photo's Claude and our bottles are paired by box IoU (Hungarian, IoU >= 0.3). An item is one physical bottle
with one or two answers.

Blind means nothing the page gets tells the systems apart: items reach it under a keyed token (the internal id says
which system answered a single-answer item), A/B is keyed the same way, both answers are written in one format
(brand and product only), and a verdict gets no reveal back. The key is JUDGE_SECRET from .env, or a fresh one per
server start (a page loaded before a restart then needs a reload). The page shows results and history only once
the judge has nothing left to judge. The only write is POST /api/judge/judgement, append-only to
data/test_bars/judgements.csv.
"""
import csv
import hashlib
import hmac
import io
import math
import os
import random
import re
import secrets
import threading
import time
from collections import defaultdict
from contextlib import suppress
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment

import datafiles as df
from boxes import iou, padded_crop
from photos import open_photo

DATA = df.DATA
CLAUDE = Path(os.environ.get("JUDGE_CLAUDE", DATA / "predictions_bars5_claude.csv"))
OURS = Path(os.environ.get("JUDGE_OURS", DATA / "predictions_bars5_siglip2_v2_ocr.csv"))
OUT = Path(os.environ.get("JUDGE_OUT", DATA / "test_bars" / "judgements.csv"))
COLS = ("photo", "item_id", "box", "claude_answer", "ours_answer", "a_is", "verdict", "judge", "timestamp")
VERDICTS = ("a", "b", "both", "neither", "cant_tell", "skip")
SETTLED = 60  # seconds: a file untouched this long is finished, so its last photo is complete
IOU_MIN = 0.3
CROP_SIDE = 1100  # px: bigger crops are shrunk to this
PHOTO_SIDE = 800  # the whole photo beside the crop, for context
TOKEN = re.compile(r"^[0-9a-f]{16}$")
JUDGE = re.compile(r"^[A-Za-z][A-Za-z .'-]{0,29}$")
UNKNOWN = "Unknown brand"

api = APIRouter(prefix="/api/judge", tags=["judge"])
router = APIRouter()
_write = threading.Lock()


@router.get("/judge", include_in_schema=False)
def judge_page() -> FileResponse:
    return FileResponse(df.WEB / "judge.html")


def _stamp(path: Path) -> tuple:
    """What items() depends on in one file: its version, and whether it may still be being written."""
    try:
        st = path.stat()
    except OSError:
        return None, None, False
    return st.st_mtime_ns, st.st_size, time.time() - st.st_mtime < SETTLED


def _rows(path: Path, run: str | None, fresh: bool) -> dict[str, list[dict]]:
    """photo name -> rows with a box. Photos still being written (last photo of a fresh file) are dropped."""
    rows = df.cached_csv(path)
    if not rows:
        return {}
    if run is None:
        run = min({r.get("run", "1") for r in rows}, key=df.run_key)
    by = defaultdict(list)
    order = []
    for r in rows:
        if r.get("run", "1") != run:
            continue
        name = Path(r.get("image", "")).name
        if not df.BAR_NAME.match(name):
            continue
        if name not in by:
            order.append(name)
        box = df.parse_bbox(r.get("bbox", ""))
        if box and box[2] > 0 and box[3] > 0:
            by[name].append(r | {"_box": box})
    if fresh and order:
        by.pop(order[-1], None)
    return by


def answer(r: dict | None) -> str:
    """The whole answer, as recorded in judgements.csv."""
    if r is None:
        return ""
    size = f"{r.get('size', '')} {r.get('unit', '')}".strip()
    if size:
        with suppress(TypeError, ValueError, KeyError):  # free text such as "33 cl" stays as it is
            size = f"{float(r['size']):g} {r.get('unit', '')}".strip()
    parts = [r.get("brand", "").strip() or UNKNOWN, r.get("product", "").strip(), size]
    return " | ".join(p for p in parts if p)


def shown(r: dict | None) -> str:
    """Brand and product only, one format for both systems: sizes, Claude's "Unknown" and product names that repeat
    the brand would give a system away."""
    if r is None:
        return ""
    brand = r.get("brand", "").strip()
    if brand.lower() in ("", "unknown"):
        brand = UNKNOWN
    product = r.get("product", "").strip()
    if brand != UNKNOWN and product.lower().startswith(brand.lower()):
        product = product[len(brand):].strip(" -,")
    return " | ".join(p for p in (brand, product) if p)


def union_box(a, b):
    if a is None or b is None:
        return a or b
    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
    x2, y2 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [x1, y1, x2 - x1, y2 - y1]


@lru_cache(maxsize=1)
def secret() -> bytes:
    """Key for item tokens and A/B, read on first use so .env is loaded by then."""
    return (os.environ.get("JUDGE_SECRET") or secrets.token_hex(16)).encode()


def keyed(text: str) -> bytes:
    return hmac.new(secret(), text.encode(), hashlib.sha256).digest()


def token(item_id: str) -> str:
    """What the page calls an item."""
    return keyed(f"id|{item_id}").hex()[:16]


def a_is(item_id: str, single: str | None) -> str:
    if single:
        return single
    return "claude" if keyed(f"ab|{item_id}")[0] % 2 == 0 else "ours"


def items() -> dict[str, dict]:
    """Every judgeable item, keyed by id. Only photos present in both files, so ids don't shift later. Built once
    per version of the two files: every /item, /crop and /queue request needs it."""
    return _items(_stamp(CLAUDE), _stamp(OURS))


def find(tok: str) -> dict:
    """The item the page calls `tok`."""
    if TOKEN.match(tok):
        for it in items().values():
            if it["token"] == tok:
                return it
    raise HTTPException(404, "no such item")


@lru_cache(maxsize=4)
def _items(claude_stamp: tuple, ours_stamp: tuple) -> dict[str, dict]:
    claude, ours = _rows(CLAUDE, "1", claude_stamp[2]), _rows(OURS, None, ours_stamp[2])
    out = {}
    for photo in sorted(set(claude) & set(ours)):
        cs, os_ = claude[photo], ours[photo]
        pairs = []
        if cs and os_:
            m = np.array([[iou(c["_box"], o["_box"]) for o in os_] for c in cs])
            ri, ci = linear_sum_assignment(-m)
            pairs = [(i, j) for i, j in zip(ri, ci, strict=True) if m[i, j] >= IOU_MIN]
        used_c, used_o = {i for i, _ in pairs}, {j for _, j in pairs}
        groups = pairs + [(i, None) for i in range(len(cs)) if i not in used_c] + [(None, j) for j in range(len(os_)) if j not in used_o]
        for i, j in groups:
            c = cs[i] if i is not None else None
            o = os_[j] if j is not None else None
            iid = f"{Path(photo).stem}-c{c['position'] if c else ''}-o{o['position'] if o else ''}"
            single = None if c and o else ("claude" if c else "ours")
            out[iid] = {"id": iid, "token": token(iid), "photo": photo,
                        "box": union_box(c and c["_box"], o and o["_box"]),
                        "claude": answer(c), "ours": answer(o), "claude_shown": shown(c), "ours_shown": shown(o),
                        "claude_brand": (c or {}).get("brand", ""), "ours_brand": (o or {}).get("brand", ""),
                        "single": single, "a_is": a_is(iid, single)}
    return out


def judgements() -> list[dict]:
    return df.cached_csv(OUT)


def queue_ids(all_items: dict[str, dict]) -> list[str]:
    """Deterministic shuffle: photos in a seeded random order, items shuffled within each photo, then dealt
    round-robin so consecutive items come from different photos."""
    per = defaultdict(list)
    for iid, it in sorted(all_items.items()):
        per[it["photo"]].append(iid)
    photos = sorted(per)
    random.Random(7).shuffle(photos)
    for p in photos:
        random.Random(f"7|{p}").shuffle(per[p])
    return [per[p][k] for k in range(max((len(v) for v in per.values()), default=0)) for p in photos if k < len(per[p])]


def public(it: dict) -> dict:
    """What the page may see before judging: no system names, nothing that tells the answers apart."""
    a, b = ("claude", "ours") if it["a_is"] == "claude" else ("ours", "claude")
    w, h = df.image_size(df.bar_photo(it["photo"]))
    return {"id": it["token"], "photo": it["photo"], "box": [round(v) for v in it["box"]], "width": w, "height": h,
            "single": bool(it["single"]), "a": it[f"{a}_shown"], "b": None if it["single"] else it[f"{b}_shown"]}


@api.get("/queue")
def queue(judge: str = ""):
    """Remaining items in judging order, as tokens. An item is done once anyone has judged it, or this judge
    skipped it."""
    all_items = items()
    done, skipped = set(), set()
    for j in judgements():
        if j.get("verdict") == "skip":
            if j.get("judge", "").lower() == judge.lower():
                skipped.add(j.get("item_id"))
        else:
            done.add(j.get("item_id"))
    order = queue_ids(all_items)
    remaining = [i for i in order if i not in done and i not in skipped]
    return {"total": len(order), "judged": sum(i in done for i in order), "remaining": len(remaining),
            "photos": len({it["photo"] for it in all_items.values()}),
            "ids": [all_items[i]["token"] for i in remaining],
            "inputs": {"claude": CLAUDE.is_file() and CLAUDE.stat().st_size > 0,
                       "ours": OURS.is_file() and OURS.stat().st_size > 0}}


@api.get("/item/{tok}")
def item(tok: str):
    return public(find(tok))


@lru_cache(maxsize=512)  # crops are under 200 KB
def _crop(path: str, box: tuple, mtime_ns: int) -> bytes:
    c, _ = padded_crop(open_photo(path), box)
    c.thumbnail((CROP_SIDE, CROP_SIDE))
    buf = io.BytesIO()
    c.save(buf, "JPEG", quality=88)
    return buf.getvalue()


@api.get("/crop/{tok}")
def crop(tok: str):
    it = find(tok)
    p = df.bar_photo(it["photo"])
    data = _crop(str(p), tuple(round(v) for v in it["box"]), p.stat().st_mtime_ns)
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@api.get("/photo/{name}")
def photo(name: str):
    return Response(df.jpeg(df.bar_photo(name), PHOTO_SIDE), media_type="image/jpeg")


class Judgement(BaseModel):
    item_id: str  # the token the page got from /queue
    verdict: str
    judge: str


@api.post("/judgement")
def post_judgement(body: Judgement):
    """Append one judgement. Everything except the verdict and the judge's name is filled in server-side. Nothing
    about which answer was which comes back: the page would learn each system's style from it."""
    if body.verdict not in VERDICTS:
        raise HTTPException(400, f"verdict must be one of {', '.join(VERDICTS)}")
    judge = body.judge.strip()
    if not JUDGE.match(judge):
        raise HTTPException(400, "judge name: letters, spaces, . ' - only, up to 30 characters")
    it = find(body.item_id)
    if it["single"] and body.verdict in ("b", "both"):
        raise HTTPException(400, "this item has only answer A")
    row = {"photo": it["photo"], "item_id": it["id"], "box": "[" + ", ".join(str(round(v)) for v in it["box"]) + "]",
           "claude_answer": it["claude"], "ours_answer": it["ours"], "a_is": it["a_is"], "verdict": body.verdict,
           "judge": judge, "timestamp": datetime.now(UTC).isoformat(timespec="seconds")}
    with _write:
        new = not OUT.is_file() or OUT.stat().st_size == 0
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            if new:
                w.writeheader()
            w.writerow(row)
    return {"ok": True}


def right(j: dict) -> dict[str, bool | None]:
    """Per system: True / False if judged on brand, None if it gave no answer or the photo can't tell."""
    v, a = j.get("verdict"), j.get("a_is")
    b = "ours" if a == "claude" else "claude"
    has = {"claude": bool(j.get("claude_answer")), "ours": bool(j.get("ours_answer"))}
    if v in ("cant_tell", "skip") or a not in ("claude", "ours"):
        return {"claude": None, "ours": None}
    ok = {a: v in ("a", "both"), b: v in ("b", "both")}
    return {s: (ok[s] if has[s] else None) for s in ("claude", "ours")}


def reveal(j: dict) -> dict:
    r = right(j)
    return {"item_id": j.get("item_id"), "photo": j.get("photo"), "a_is": j.get("a_is"), "verdict": j.get("verdict"),
            "judge": j.get("judge"), "timestamp": j.get("timestamp"),
            "claude": {"answer": j.get("claude_answer", ""), "right": r["claude"]},
            "ours": {"answer": j.get("ours_answer", ""), "right": r["ours"]}}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def latest() -> list[dict]:
    """One judgement per item (the latest non-skip), so a re-judged item counts once."""
    by = {}
    for j in judgements():
        if j.get("verdict") in VERDICTS and j.get("verdict") != "skip":
            by[j.get("item_id")] = j
    return list(by.values())


def stats() -> dict:
    js = latest()
    out = {"judged": len(js), "cant_tell": sum(j["verdict"] == "cant_tell" for j in js), "systems": {}}
    for s in ("claude", "ours"):
        has = [j for j in js if j.get(f"{s}_answer")]
        scored = [right(j)[s] for j in has]
        scored = [x for x in scored if x is not None]
        k, n = sum(scored), len(scored)
        lo, hi = wilson(k, n)
        out["systems"][s] = {"answered": len(has), "scored": n, "right": k,
                             "accuracy": k / n if n else None, "ci_low": lo, "ci_high": hi}
    paired = [j for j in js if j.get("claude_answer") and j.get("ours_answer") and j["verdict"] != "cant_tell"]
    out["paired"] = {"n": len(paired), **{v: sum(j["verdict"] == v for j in paired) for v in ("both", "neither")},
                     "claude_only": sum(right(j)["claude"] and not right(j)["ours"] for j in paired),
                     "ours_only": sum(right(j)["ours"] and not right(j)["claude"] for j in paired)}
    out["paired"]["agreement"] = (out["paired"]["both"] + out["paired"]["neither"]) / len(paired) if paired else None
    return out


@api.get("/results")
def results():
    """Live accuracy per system plus how often the two systems agree, by name and by the judge's verdict."""
    s = stats()
    all_items = items().values()
    pairs = [it for it in all_items if not it["single"]]
    same = sum(bool(_norm(it["claude_brand"])) and _norm(it["claude_brand"]) == _norm(it["ours_brand"]) for it in pairs)
    s["items"] = {"total": len(all_items), "paired": len(pairs),
                  "claude_only": sum(it["single"] == "claude" for it in all_items),
                  "ours_only": sum(it["single"] == "ours" for it in all_items),
                  "brand_agreement": same / len(pairs) if pairs else None}
    return s


@api.get("/history")
def history(limit: int = 50):
    """Latest judgements first, with which answer was which revealed."""
    limit = max(1, min(limit, 500))
    return [reveal(j) for j in reversed(judgements()[-limit:])]


def leaderboard_rows() -> list[dict]:
    """Two Testing-leaderboard rows (human-judged brand accuracy), or none when nothing is judged yet."""
    s = stats()
    rows = []
    # named after the files, since JUDGE_CLAUDE / JUDGE_OURS can point anywhere
    names = {"claude": (f"Claude baseline, open-ended ({CLAUDE.stem.removeprefix('predictions_')})", CLAUDE.name),
             "ours": (f"Ours: {OURS.stem.removeprefix('predictions_')}", OURS.name)}
    for k, (model, f) in names.items():
        m = s["systems"][k]
        if not m["scored"]:
            continue
        rows.append({"date": "", "experiment": f"Bar photos, human-judged (n={m['scored']})", "model": model,
                     "predictions_file": f, "inputs": "bar photos", "valid": True, "baseline": False,
                     "notes": f"Blind A/B, brand right by human judgement, can't-tell excluded. 95% Wilson "
                              f"{100 * m['ci_low']:.0f} to {100 * m['ci_high']:.0f}%. {m['right']} of {m['scored']} right.",
                     "brand_recall": m["accuracy"], "recall": None, "recall_sd": None, "bar_recall": None,
                     "precision": None, "count_error": None, "bottles_found": None})
    return rows


router.include_router(api)
