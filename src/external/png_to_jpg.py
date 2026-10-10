"""Convert every .png under a directory to .jpg (quality 95) in place, deleting the PNG.
Used for whiskeyClassification (22.6 GB of PNG) to keep it under ~20 GB on disk.
Usage: python src/external/png_to_jpg.py data/external/whiskey-classification/dataset
"""
import sys, os
from multiprocessing import Pool
from PIL import Image

def conv(p):
    out = p[:-4] + ".jpg"
    try:
        Image.open(p).convert("RGB").save(out, quality=95)
        os.remove(p)
    except Exception as e:
        return f"{p}: {e}"

if __name__ == "__main__":
    root = sys.argv[1]
    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(".png")]
    with Pool(8) as pool:
        errs = [e for e in pool.imap_unordered(conv, files, chunksize=16) if e]
    print(f"converted {len(files) - len(errs)}, errors {len(errs)}")
    for e in errs[:20]: print(e)
