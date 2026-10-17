"""Identify every stock bottle in each photo with Claude vision. This is the baseline to beat.

Usage: python src/identify.py [photos_dir] [--runs N] [--out predictions.csv]   (defaults: data/raw, 1, data/predictions.csv)
"""
import argparse
import base64
import csv
import io
import sys
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel

from photos import EXTS, open_photo
from vlm import ALIASES, MAX_TOKENS, Refusal

MODEL = ALIASES["opus"]
SEND_SIDE = 1568  # long side of the photo sent; the baseline was measured at this size, so keep it

PROMPT = """You are cataloguing bar stock from a photo of a group of bottles. List every stock bottle or container of food or drink you can see (ignore reusable water bottles, shakers, cups and glasses),
ordered left to right (front row before back row if there are rows). For each bottle:
- position: 1-based index in that order, plus a short visual description so a human can match it (e.g. "tall green, red cap").
- bbox: [x, y, width, height] in pixels of this image, x and y at the top-left corner of the bottle.
- brand and product: read them off the label, including the exact variant (e.g. Black Label, not just Johnnie Walker); if unreadable, give your best guess.
- liquid: what is inside (e.g. vodka, gin, whisky, red wine, olive oil, syrup).
- size and unit: from the label if printed (ml for liquids, g where the label gives weight), else infer from bottle shape and typical sizes (50, 350, 500, 700, 750, 1000, 1500, 2000 ml).
- glass: clear, green, amber, frosted or opaque.
- fill_ml: ml of liquid left, from the liquid line and the bottle's shape (a neck holds less than the body). Always ml, even when size is in g or pint. null when fill_visible is false.
- id_source: "label" if you read brand and product off the label, "shape" if you guessed them from shape, colour or cap.
- fill_visible: true if you can see the liquid line through the container."""


class Bottle(BaseModel):
    position: int
    description: str
    bbox: list[int]
    brand: str
    product: str
    category: Literal["spirit", "liqueur", "wine", "beer", "mixer", "soft_drink", "syrup", "sauce", "dairy", "oil", "other"]
    liquid: str
    size: float
    unit: Literal["ml", "g", "pint"]
    glass: Literal["clear", "green", "amber", "frosted", "opaque"]
    fill_ml: float | None
    id_source: Literal["label", "shape"]
    fill_visible: bool


class Photo(BaseModel):
    bottles: list[Bottle]


PRED_FIELDS = ["run", "image", *Bottle.model_fields]  # the predictions CSV every pipeline writes


def load_image(path: Path) -> tuple[str, float]:
    """JPEG base64 for the API, plus the factor that scales its pixels back to the original photo."""
    img = open_photo(path)
    width = img.width
    img.thumbnail((SEND_SIDE, SEND_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode(), width / img.width


def identify(client: anthropic.Anthropic, path: Path) -> list[Bottle]:
    data, scale = load_image(path)
    response = client.messages.parse(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": "medium"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        output_format=Photo,
    )
    if response.stop_reason == "refusal":
        raise Refusal(f"declined to answer for {path.name}")
    bottles = response.parsed_output.bottles
    for b in bottles:
        b.bbox = [round(v * scale) for v in b.bbox]
    return bottles


def main(folder: str, runs: int, out: str = "data/predictions.csv") -> None:
    client = anthropic.Anthropic()
    images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)
    out = Path(out)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        writer.writeheader()
        for run, path in ((r, p) for r in range(1, runs + 1) for p in images):
            try:
                bottles = identify(client, path)
            except (anthropic.APIError, Refusal) as e:
                print(f"FAIL {path.name}: {e}", file=sys.stderr)
                continue
            print(f"run {run} {path.name}: {len(bottles)} bottles")
            for b in bottles:
                writer.writerow({"run": run, "image": path.name, **b.model_dump()})
                fill = f"{b.fill_ml:g} ml left" if b.fill_ml is not None else "fill not visible"
                print(f"  {b.position}. {b.brand} {b.product}, {b.size:g}{b.unit}, {b.liquid}, {fill} (id from {b.id_source})")
    print(f"\n{len(images)} images x {runs} runs -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("photos", nargs="?", default="data/raw")
    ap.add_argument("--runs", type=int, default=1, help="repeat runs, to measure the spread")
    ap.add_argument("--out", default="data/predictions.csv")
    args = ap.parse_args()
    main(args.photos, args.runs, args.out)
