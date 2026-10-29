"""Download the bar test photos that are missing from data/test_bars/img/.

33 of the web photos in data/test_bars/manifest.csv have no known licence, so the
repo does not ship them. This script fetches each missing photo from the manifest's
image_url, scales it to the manifest's width x height (the size the labels were made
on) and saves it as JPEG. Files that already exist are skipped. Each photo is checked
against the manifest's perceptual hash; a distance over 6 means the source changed and
the labels may not fit any more. Check each site's terms before using the photos.

usage: python scripts/download_bars.py [bar_01.jpg bar_02.jpg ...]   (default: all missing)
"""
import contextlib
import csv
import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

import imagehash
from PIL import Image

BARS = Path(__file__).resolve().parents[1] / "data" / "test_bars"
UA = "backbar-vision-download/1.0"  # some hosts return 403 to "Mozilla/5.0 (compatible; ...)"
MAX_PHASH_DIST = 6


def fetch(url, tries=2):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for _ in range(tries - 1):
        # slow hosts (pxhere) sometimes stall once
        with contextlib.suppress(TimeoutError, urllib.error.URLError), urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def to_manifest_size(im, w, h):
    im = im.convert("RGB")
    if im.size == (w, h):
        return im
    if abs(im.width / im.height - w / h) > 0.01:
        raise ValueError(f"aspect ratio changed: got {im.width}x{im.height}, manifest {w}x{h}")
    return im.resize((w, h), Image.LANCZOS)


def main(wanted):
    with open(BARS / "manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if wanted:
        unknown = set(wanted) - {r["file"] for r in rows}
        if unknown:
            sys.exit(f"not in manifest: {', '.join(sorted(unknown))}")
        rows = [r for r in rows if r["file"] in wanted]
    out_dir = BARS / "img"
    out_dir.mkdir(parents=True, exist_ok=True)
    got, skipped, failed = [], [], []
    for r in rows:
        dest = out_dir / r["file"]
        if dest.exists():
            skipped.append(r["file"])
            continue
        try:
            im = Image.open(io.BytesIO(fetch(r["image_url"])))
            im = to_manifest_size(im, int(r["width"]), int(r["height"]))
            dist = imagehash.phash(im) - imagehash.hex_to_hash(r["phash"])
            if dist > MAX_PHASH_DIST:
                raise ValueError(f"image differs from the labelled one (phash distance {dist})")
            im.save(dest, "JPEG", quality=95)
            got.append(r["file"])
            print(f"ok      {r['file']}")
        except (OSError, ValueError) as e:  # report and carry on: one dead link should not stop the rest
            failed.append(r["file"])
            print(f"FAILED  {r['file']}: {e}  ({r['image_url']})")
    print(f"\n{len(got)} downloaded, {len(skipped)} already present, {len(failed)} failed")
    if failed:
        print("failed: " + " ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
