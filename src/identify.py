"""Identify every stock bottle in each photo with Claude vision. This is the baseline to beat.

Usage: python src/identify.py data/raw [runs]     -> writes data/predictions.csv
"""
import base64
import csv
import io
import sys
from pathlib import Path
from typing import Literal

import anthropic
from dotenv import load_dotenv
from PIL import Image
from pillow_heif import register_heif_opener
from pydantic import BaseModel

register_heif_opener()
load_dotenv()

MODEL = "claude-opus-5-5"
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp"}

PROMPT = """You are cataloguing bar stock from a photo of a group of bottles. List every stock bottle or container of food or drink you can see (ignore reusable water bottles, shakers, cups and glasses),
ordered left to right (front row before back row if there are rows). For each bottle:
- position: 1-based index in that order, plus a short visual description so a human can match it (e.g. "tall green, red cap").
- brand and product: read them off the label; if unreadable, give your best guess and lower confidence.
- size and unit: from the label if printed (ml for liquids, g where the label gives weight), else infer from bottle shape and typical sizes (50, 350, 500, 700, 750, 1000, 1500, 2000 ml).
- liquid: what is inside (e.g. vodka, gin, whisky, red wine, olive oil, syrup).
- glass: clear, green, amber, frosted or opaque.
- confidence: 0 to 1, your overall confidence in brand + size + liquid."""


class Bottle(BaseModel):
    position: int
    description: str
    brand: str
    product: str
    category: Literal["spirit", "liqueur", "wine", "beer", "mixer", "soft_drink", "syrup", "sauce", "dairy", "oil", "other"]
    size: float
    unit: Literal["ml", "g", "pint"]
    liquid: str
    glass: Literal["clear", "green", "amber", "frosted", "opaque"]
    confidence: float


class Photo(BaseModel):
    bottles: list[Bottle]


def load_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def identify(client: anthropic.Anthropic, path: Path) -> list[Bottle]:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        output_config={"effort": "medium"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": load_image(path)}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        output_format=Photo,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"refused: {path.name}")
    return response.parsed_output.bottles


def main(folder: str, runs: int) -> None:
    client = anthropic.Anthropic()
    images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)
    out = Path("data/predictions.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "image", *Bottle.model_fields])
        writer.writeheader()
        for run, path in ((r, p) for r in range(1, runs + 1) for p in images):
            try:
                bottles = identify(client, path)
            except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError) as e:
                print(f"FAIL {path.name}: {e}", file=sys.stderr)
                continue
            print(f"run {run} {path.name}: {len(bottles)} bottles")
            for b in bottles:
                writer.writerow({"run": run, "image": path.name, **b.model_dump()})
                print(f"  {b.position}. {b.brand} {b.product}, {b.size:g}{b.unit}, {b.liquid} ({b.confidence:.2f})")
    print(f"\n{len(images)} images x {runs} runs -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw", int(sys.argv[2]) if len(sys.argv) > 2 else 1)
