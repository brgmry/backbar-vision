"""Open-weight VLM baseline, run locally with MLX. Same prompt and output schema as identify.py.

Usage: python src/identify_vlm.py --model mlx-community/Qwen3-VL-8B-Instruct-8bit [--runs 1] [--folder data/raw] [--out ...]
       -> data/predictions_vlm_<model-short>.csv
"""
import argparse
import csv
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image
from pillow_heif import register_heif_opener
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent))
from identify import EXTS, PROMPT, Bottle  # noqa: E402

register_heif_opener()

MAX_SIDE = 1568  # same as the Claude baseline

FIELDS = list(Bottle.model_fields)
SCHEMA_HINT = """

Answer with JSON only, no prose, no markdown fences, in exactly this shape:
{"bottles": [{"position": 1, "description": "...", "bbox": [x, y, w, h], "brand": "...", "product": "...",
"category": "spirit|liqueur|wine|beer|mixer|soft_drink|syrup|sauce|dairy|oil|other", "liquid": "...",
"size": 700, "unit": "ml|g|pint", "glass": "clear|green|amber|frosted|opaque", "fill_ml": 350 or null,
"id_source": "label|shape", "fill_visible": true}]}
The image is %dx%d pixels."""

ENUMS = {
    "category": ["spirit", "liqueur", "wine", "beer", "mixer", "soft_drink", "syrup", "sauce", "dairy", "oil", "other"],
    "unit": ["ml", "g", "pint"],
    "glass": ["clear", "green", "amber", "frosted", "opaque"],
    "id_source": ["label", "shape"],
}


def short_name(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.split("/")[-1].lower()).strip("-")


def prep_image(path: Path, tmpdir: Path) -> tuple[Path, float, tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    width = img.width
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = tmpdir / f"{path.stem}.jpg"
    img.save(out, "JPEG", quality=90)
    return out, width / img.width, img.size


def extract_json(text: str):
    """Best-effort: strip thinking and fences, take the outermost object or array, repair truncation."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    body = text[start:]
    try:
        return json.JSONDecoder().raw_decode(body)[0]
    except json.JSONDecodeError:
        pass
    # truncated output: keep every complete bottle object
    objs = []
    for m in re.finditer(r"\{[^{}]*\}", body):
        try:
            objs.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return {"bottles": objs} if objs else None


def coerce(raw: dict, pos: int) -> Bottle | None:
    d = {k: raw.get(k) for k in FIELDS}
    d["position"] = d["position"] if isinstance(d["position"], int) else pos
    for k in ("description", "brand", "product", "liquid"):
        d[k] = str(d[k] or "")
    bbox = d["bbox"] if isinstance(d["bbox"], list) else []
    try:
        d["bbox"] = [int(float(v)) for v in bbox[:4]] + [0] * (4 - len(bbox[:4]))
    except (TypeError, ValueError):
        d["bbox"] = [0, 0, 0, 0]
    for k, allowed in ENUMS.items():
        v = str(d[k] or "").lower().strip()
        d[k] = v if v in allowed else ("other" if k == "category" else allowed[0] if k != "glass" else "clear")
    try:
        d["size"] = float(re.sub(r"[^0-9.]", "", str(d["size"])) or 0)
    except ValueError:
        d["size"] = 0.0
    try:
        d["fill_ml"] = None if d["fill_ml"] in (None, "", "null") else float(d["fill_ml"])
    except (TypeError, ValueError):
        d["fill_ml"] = None
    d["fill_visible"] = d["fill_visible"] in (True, "true", "True", 1)
    if not d["fill_visible"]:
        d["fill_ml"] = None
    try:
        return Bottle(**d)
    except ValidationError:
        return None


def parse(text: str) -> list[Bottle]:
    data = extract_json(text)
    if isinstance(data, dict):
        data = data.get("bottles", [])
    if not isinstance(data, list):
        return []
    out = [coerce(r, i) for i, r in enumerate(data, 1) if isinstance(r, dict)]
    return [b for b in out if b is not None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-8bit")
    ap.add_argument("--folder", default="data/raw")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out")
    ap.add_argument("--from-raw", action="store_true", help="re-parse saved outputs in runs/vlm_raw, no generation")
    args = ap.parse_args()

    if not args.from_raw:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template

        model, processor = load(args.model)
    # Qwen3-VL and Qwen3.5 ground boxes on a 0-1000 grid whatever the prompt says
    norm1000 = "qwen3" in args.model.lower()
    out = Path(args.out or f"data/predictions_vlm_{short_name(args.model)}.csv")
    raw_dir = Path(f"runs/vlm_raw/{short_name(args.model)}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS)
    times = []
    with tempfile.TemporaryDirectory() as tmp, out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "image", *FIELDS])
        writer.writeheader()
        for run, path in ((r, p) for r in range(1, args.runs + 1) for p in images):
            img_path, scale, (w, h) = prep_image(path, Path(tmp))
            raw_file = raw_dir / f"run{run}_{path.stem}.txt"
            t0 = time.time()
            if args.from_raw:
                text = raw_file.read_text()
            else:
                prompt = apply_chat_template(processor, model.config, PROMPT + SCHEMA_HINT % (w, h), num_images=1)
                result = generate(model, processor, prompt, image=[str(img_path)], max_tokens=args.max_tokens,
                                  temperature=args.temperature, verbose=False)
                text = result.text if hasattr(result, "text") else str(result)
                raw_file.write_text(text)
            dt = time.time() - t0
            times.append(dt)
            bottles = parse(text)
            print(f"run {run} {path.name}: {len(bottles)} bottles in {dt:.0f}s", flush=True)
            for b in bottles:
                if norm1000:  # [x1, y1, x2, y2] on 0-1000 -> [x, y, w, h] in the sent image
                    x1, y1, x2, y2 = b.bbox[0] * w / 1000, b.bbox[1] * h / 1000, b.bbox[2] * w / 1000, b.bbox[3] * h / 1000
                    b.bbox = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]
                b.bbox = [round(v * scale) for v in b.bbox]
                writer.writerow({"run": run, "image": path.name, **b.model_dump()})
            f.flush()
    print(f"\n{len(images)} images x {args.runs} runs -> {out}; mean {sum(times) / len(times):.1f}s per photo")


if __name__ == "__main__":
    main()
