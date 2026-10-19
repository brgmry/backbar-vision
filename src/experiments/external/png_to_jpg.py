"""Convert every .png under a directory to .jpg (quality 95) in place, deleting the PNG.
Used for whiskeyClassification (22.6 GB of PNG) to keep it under ~20 GB on disk.
Usage: python src/experiments/external/png_to_jpg.py data/external/whiskey-classification/dataset
"""
import os
import sys
from multiprocessing import Pool

from PIL import Image


def conv(p):
    out = p[:-4] + ".jpg"
    try:
        with Image.open(p) as im:
            im.convert("RGB").save(out, quality=95)
        os.remove(p)
    except (OSError, ValueError, Image.DecompressionBombError) as e:  # reported at the end; the PNG stays
        return f"{p}: {e}"
    return None


if __name__ == "__main__":
    root = sys.argv[1]
    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(".png")]
    with Pool(8) as pool:
        errs = [e for e in pool.imap_unordered(conv, files, chunksize=16) if e]
    print(f"converted {len(files) - len(errs)}, errors {len(errs)}")
    for e in errs[:20]:
        print(e)
