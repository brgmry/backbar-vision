"""Pre-label every crop with its nearest packshot in data/reference, for a human to correct.

Writes data/crops/labels.csv (photo, crop, item_id, sim) and a contact sheet per photo in
data/crops/sheets/. item_id 0 means "not a stock item". Edit labels.csv by hand, it is the ground truth.
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from embed import embed

refs = sorted(Path("data/reference").glob("[0-9]*-*.*"))
ref_ids = [int(p.name.split("-")[0]) for p in refs]
boxes = list(csv.DictReader(open("data/crops/boxes.csv")))
crops = [Path(f"data/crops/{b['photo']}/{int(b['crop']):02d}.jpg") for b in boxes]

sims = embed(crops) @ embed(refs).T
best = sims.max(dim=1)

rows = [{"photo": b["photo"], "crop": b["crop"], "item_id": ref_ids[i], "sim": round(float(s), 3)}
        for b, s, i in zip(boxes, best.values, best.indices)]
with open("data/crops/labels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

sheets = Path("data/crops/sheets")
sheets.mkdir(exist_ok=True)
for photo in sorted({r["photo"] for r in rows}):
    mine = [(r, c) for r, c in zip(rows, crops) if r["photo"] == photo]
    cell, cols = 220, 8
    sheet = Image.new("RGB", (cell * cols, (cell + 30) * ((len(mine) + cols - 1) // cols)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (r, c) in enumerate(mine):
        im = Image.open(c)
        im.thumbnail((cell, cell))
        x, y = (k % cols) * cell, (k // cols) * (cell + 30)
        sheet.paste(im, (x, y))
        d.text((x + 4, y + cell + 4), f"#{r['crop']} -> {r['item_id']} ({r['sim']})", fill="black")
    sheet.save(sheets / f"{photo}.jpg", quality=85)
print(f"{len(rows)} crops labelled -> data/crops/labels.csv")
