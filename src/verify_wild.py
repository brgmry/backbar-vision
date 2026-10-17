"""Clean the in-the-wild brand photos for free: keep a photo only if its own label text (Apple Vision OCR)
names the brand it was searched for. Turns search-query labels into verified real-world brand photos.

Usage: python src/verify_wild.py [workers]   -> data/external/wild/verified.csv (file, brand, brand_key, text)
"""
import argparse
import csv
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from rapidfuzz import fuzz

from train_index import key

WILD = Path("data/external/wild")
MIN_CONFIDENCE = 0.4  # OCR lines Apple Vision is less sure of are ignored
MIN_WORD = 4  # brand words shorter than this ("the", "de") prove nothing
FUZZY_WORD = 6  # words this long may match fuzzily (OCR slips); shorter ones must appear as a whole word
BRAND_MATCH = 90  # fuzzy partial_ratio a long brand word needs inside the label text
TEXT_KEPT = 200  # characters of label text written to verified.csv


def check(row):
    from ocr import read_text  # macOS only; unreadable files come back with no text
    text = key(" ".join(t for t, _ in read_text(WILD / row["file"], min_confidence=MIN_CONFIDENCE)))
    b = key(row["brand"])
    # the brand's main word(s) must appear; short brands need an exact word to avoid false hits
    words = [w for w in b.split() if len(w) >= MIN_WORD] or [b]
    ok = any((fuzz.partial_ratio(w, text) >= BRAND_MATCH if len(w) >= FUZZY_WORD else w in text.split())
             for w in words)
    return {"file": row["file"], "brand": row["brand"], "brand_key": b, "text": text[:TEXT_KEPT]} if ok else None


def main(workers: int):
    with open(WILD / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    with Pool(workers) as pool:
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
    c = Counter(r["brand_key"] for r in kept)
    print(f"verified {len(kept)}/{len(rows)} photos, {len(c)} brands, "
          f"median {sorted(c.values())[len(c) // 2] if c else 0} per brand")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("workers", nargs="?", type=int, default=8)
    main(ap.parse_args().workers)
