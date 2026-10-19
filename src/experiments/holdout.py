"""The bar test set stays out of training: no training image may be a near copy of a test photo."""
import re
import sys
from pathlib import Path

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TEST_HASHES = ROOT / "data/test_bars/phashes.txt"
MAX_PHASH_DIST = 6  # at or under this pHash distance two images are the same photo, resized or recompressed
READ_ERRORS = (OSError, ValueError, Image.DecompressionBombError)


def test_hashes() -> list[imagehash.ImageHash]:
    """pHashes of the test photos. Stops when there are none: an empty list would hold nothing out."""
    text = TEST_HASHES.read_text() if TEST_HASHES.exists() else ""
    hashes = [imagehash.hex_to_hash(m.group(0)) for m in re.finditer(r"\b[0-9a-f]{16}\b", text)]
    if not hashes:
        sys.exit(f"no test photo hashes in {TEST_HASHES}: stopping, since nothing would be held out")
    return hashes


def test_distance(h: imagehash.ImageHash, test: list[imagehash.ImageHash]) -> int:
    """pHash distance to the closest test photo."""
    return min(h - t for t in test)


def image_phash(path) -> imagehash.ImageHash | None:
    """pHash of an image file, as the test hashes were made (from RGB); None when the file can't be read."""
    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB"))
    except READ_ERRORS:
        return None
