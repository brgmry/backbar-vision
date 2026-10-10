"""Identify bottles in photos with Claude vision. This is the baseline to beat.

Usage: python src/identify.py data/raw            -> writes data/predictions.csv
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

PROMPT = """You are cataloguing bar stock from a photo. Identify the main bottle in the image.
- brand and product: read them off the label; if unreadable, give your best guess and lower confidence.
- size_ml: from the label if printed, else infer from bottle shape and typical sizes (50, 200, 350, 500, 700, 750, 1000, 1750).
- liquid: what is inside (e.g. vodka, gin, whisky, red wine, olive oil, syrup).
- glass: clear, green, amber, frosted or opaque.
- confidence: 0 to 1, your overall confidence in brand + size + liquid."""


class Bottle(BaseModel):
    brand: str
    product: str
    bottle_type: Literal["spirit", "wine", "beer", "liqueur", "soft_drink", "syrup", "oil", "other"]
    size_ml: int
    liquid: str
    glass: Literal["clear", "green", "amber", "frosted", "opaque"]
    confidence: float


def load_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((1568, 1568))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def identify(client: anthropic.Anthropic, path: Path) -> Bottle:
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
        output_format=Bottle,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"refused: {path.name}")
    return response.parsed_output


def main(folder: str) -> None:
    client = anthropic.Anthropic()
    images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)
    out = Path("data/predictions.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", *Bottle.model_fields])
        writer.writeheader()
        for path in images:
            try:
                bottle = identify(client, path)
            except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError) as e:
                print(f"FAIL {path.name}: {e}", file=sys.stderr)
                continue
            writer.writerow({"image": path.name, **bottle.model_dump()})
            print(f"{path.name}: {bottle.brand} {bottle.product}, {bottle.size_ml}ml, {bottle.liquid} ({bottle.confidence:.2f})")
    print(f"\n{len(images)} images -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
