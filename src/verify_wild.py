"""Clean the in-the-wild brand photos for free: keep a photo only if its own label text (Apple Vision OCR)
names the brand it was searched for. Turns search-query labels into verified real-world brand photos.

Usage: python src/verify_wild.py [workers]   -> data/external/wild/verified.csv (file, brand, brand_key, text)
"""
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

from rapidfuzz import fuzz

from train_index import key

WILD = Path("data/external/wild")


def check(row):
    from ocr import read_text
    try:
        text = key(" ".join(t for t, _ in read_text(WILD / row["file"], min_confidence=0.4)))
    except Exception:
        return None
    b = key(row["brand"])
    # the brand's main word(s) must appear; short brands need an exact word to avoid false hits
    words = [w for w in b.split() if len(w) >= 4] or [b]
    ok = any((fuzz.partial_ratio(w, text) >= 90 if len(w) >= 6 else w in text.split()) for w in words)
    return {"file": row["file"], "brand": row["brand"], "brand_key": b, "text": text[:200]} if ok else None


if __name__ == "__main__":
    rows = list(csv.DictReader(open(WILD / "manifest.csv")))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        kept = []
        for n, r in enumerate(pool.imap_unordered(check, rows, chunksize=16), 1):
            if r:
                kept.append(r)
            if n % 1000 == 0:
                print(f"{n}/{len(rows)} checked, {len(kept)} verified", flush=True)
    with open(WILD / "verified.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "brand", "brand_key", "text"])
        w.writeheader()
        w.writerows(kept)
    from collections import Counter
    c = Counter(r["brand_key"] for r in kept)
    print(f"verified {len(kept)}/{len(rows)} photos, {len(c)} brands, median {sorted(c.values())[len(c)//2] if c else 0} per brand")
