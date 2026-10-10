"""HTTP API over both pipelines: one photo in, the shared output schema out. Web UI at /.

Usage (from the repo root, data/ paths are relative):
    .venv/bin/uvicorn api:app --app-dir src
    curl -F image=@photo.jpg 'localhost:8000/identify?pipeline=local&fill=geometry:opus'

Fill runs on alcohol bottles only (other rows keep fill null): crop clearing (qc.GeometryQC) first, then the
fill estimator named by `fill` (see fill.py; `none` skips it).
"""
import io
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import cache
from pathlib import Path
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

os.environ.setdefault("CROPS", "data/crops_whole")  # whole-bottle reference crops (src/recut_crops.py), read by match.py
import identify  # also registers the HEIC opener and loads .env

K = 3  # reference crops per item, the best k in match.py's eval
WEB = Path(__file__).resolve().parent.parent / "web"
PREVIEW_SIDE = 2400
FILL_WORKERS = 8  # concurrent Claude calls per photo

app = FastAPI(title="bottle-vision")
_local_lock = threading.Lock()  # one torch inference at a time


class BottleOut(identify.Bottle):
    glass: Literal["clear", "green", "amber", "frosted", "opaque"] | None  # the local pipeline doesn't read glass
    id_source: Literal["label", "shape", "reference"]
    fill_note: str | None = None  # why fill is null: crop rejected, not alcohol, line not visible
    label_text: str | None = None  # what Apple Vision OCR reads on the label (macOS only), not yet used for the ID


class PhotoOut(BaseModel):
    image: str
    pipeline: Literal["local", "claude"]
    bottles: list[BottleOut]


@cache
def local_models():
    """OWLv2 detector plus a DINOv2 gallery of the labelled crops from every photo. Slow the first time."""
    import detect
    import match  # embeds every labelled crop on import
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    processor = Owlv2Processor.from_pretrained(detect.MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(detect.MODEL).to(detect.DEVICE).eval()
    return processor, model, match.gallery(exclude="", k=K)


def identify_local(path: Path) -> list[BottleOut]:
    """detect.py + match.py on one photo: find bottles, match each crop to its nearest reference crop."""
    import detect
    import match
    from embed import embed

    with _local_lock, tempfile.TemporaryDirectory() as tmp:
        processor, model, gallery = local_models()
        img = detect.load(path)
        scale = Image.open(path).width / img.width
        boxes, crops = [], []
        for n, (_, (x1, y1, x2, y2)) in enumerate(detect.detect(model, processor, img, prefer_whole=True)):
            box = [max(0, x1), max(0, y1), min(img.width, x2), min(img.height, y2)]
            crop = Path(tmp) / f"{n:02d}.jpg"
            img.crop(box).save(crop, quality=92)  # same encoding as the reference crops
            boxes.append(box)
            crops.append(crop)
        if not crops:
            return []
        sims = embed(crops) @ match.vecs[gallery].T

    best = {}  # item_id -> (sim, box): one bottle per item, keep the strongest crop
    for row, box in enumerate(boxes):
        s, j = sims[row].max(dim=0)
        item = match.crops[gallery[int(j)]]["item_id"] if float(s) >= match.MIN_SIM else 0
        if item and float(s) > best.get(item, (-1,))[0]:
            best[item] = (float(s), box)

    bottles = []
    for pos, (item, (s, (x1, y1, x2, y2))) in enumerate(sorted(best.items(), key=lambda kv: kv[1][1][0]), 1):
        inv = match.inventory[item]
        bottles.append(BottleOut(
            position=pos, description=f"{s:.0%} similar to a reference crop", bbox=[round(v * scale) for v in (x1, y1, x2 - x1, y2 - y1)],
            brand=inv["brand"], product=inv["product"], category=inv["category"], liquid=inv["liquid"],
            size=inv["size"], unit=inv["unit"], glass=None, fill_ml=None, id_source="reference", fill_visible=False,
        ))
    from refine import refine  # grow boxes to the SAM 3 bottle outline: clear bases and necks OWLv2 stops short of
    with _local_lock:
        for b, box in zip(bottles, refine(Image.open(path).convert("RGB"), [b.bbox for b in bottles])):
            b.bbox = box
    return bottles


@cache
def claude_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def identify_claude(path: Path) -> list[BottleOut]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set in .env")
    try:
        bottles = identify.identify(claude_client(), path)
    except RuntimeError as e:  # refusal
        raise HTTPException(422, str(e))
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
        raise HTTPException(502, f"Claude API: {e}")
    return [BottleOut(**b.model_dump()) for b in bottles]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/preview")
def preview(image: UploadFile) -> Response:
    """The photo as a JPEG any browser can show (HEIC included), plus the original size the bboxes refer to."""
    try:
        img = Image.open(image.file)
        width, height = img.size
        img = img.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, f"{image.filename} is not an image")
    img.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"X-Width": str(width), "X-Height": str(height)})


def add_fill(photo: Image.Image, bottles: list[BottleOut], estimator) -> None:
    """Fill level for each alcohol bottle, in place: re-crop at full resolution, clear the crop, estimate.

    SAM (crop check) runs under _local_lock one bottle at a time; each cleared bottle's estimate is handed to a
    thread pool straight away, so the Claude calls run in parallel with each other and with the next bottle's SAM.
    The estimate's own SAM call is a cache hit (same crop and box as the check), so it never touches the GPU."""
    from experiment import ALCOHOL, padded_crop
    from qc import GeometryQC

    def estimate(b: BottleOut, crop: Image.Image, box: list[float], item: dict) -> None:
        with nullcontext() if remote_only(estimator) else _local_lock:
            try:
                f = estimator.estimate(crop, box, item)
            except (anthropic.APIStatusError, anthropic.APIConnectionError, RuntimeError) as e:
                b.fill_note = f"fill failed: {e}"
                return
        b.fill_ml, b.fill_visible = f.fill_ml, f.visible
        b.fill_note = None if f.visible else "liquid line not visible"

    with ThreadPoolExecutor(max_workers=FILL_WORKERS) as pool:
        jobs = []
        for b in bottles:
            if b.category not in ALCOHOL:
                b.fill_note = "not alcohol"
                continue
            crop, box = padded_crop(photo, b.bbox)
            item = b.model_dump()
            with _local_lock:
                verdict = GeometryQC().check(crop, box, item, None)
            if not verdict.ok:
                b.fill_ml, b.fill_visible, b.fill_note = None, False, f"crop rejected: {verdict.reason}"
                continue
            jobs.append(pool.submit(estimate, b, crop, box, item))
        for job in jobs:
            job.result()


def remote_only(estimator) -> bool:
    """True when estimate() does no torch work of its own: the VLM estimators (geometry's mask is already cached)."""
    from fill import Geometry, VLMFill, VLMLine

    return isinstance(estimator, VLMFill) or (isinstance(estimator, Geometry) and isinstance(estimator.line, VLMLine))


def add_label_text(photo: Image.Image, bottles: list[BottleOut]) -> None:
    """OCR on each bottle's full-resolution crop (src/ocr.py), in place. Skipped off macOS."""
    try:
        from ocr import read_text
    except ImportError:
        return
    from experiment import padded_crop

    with tempfile.TemporaryDirectory() as tmp:
        for b in bottles:
            crop_path = Path(tmp) / f"{b.position}.jpg"
            padded_crop(photo, b.bbox)[0].save(crop_path, quality=92)
            b.label_text = " | ".join(t for t, _ in read_text(crop_path)) or None


@app.post("/identify", response_model=PhotoOut)
def identify_photo(image: UploadFile, pipeline: Literal["local", "claude"] = "local",
                   fill: str = "geometry:opus", ocr: bool = True) -> PhotoOut:
    """One photo of bottles -> every stock bottle, left to right, with fill level for alcohol bottles.
    pipeline=local needs no API key; fill=none skips fill, which then needs no API key either."""
    name = Path(image.filename or "").name or "upload"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(image.file.read())
        try:
            Image.open(path)
        except UnidentifiedImageError:
            raise HTTPException(400, f"{name} is not an image")
        bottles = identify_local(path) if pipeline == "local" else identify_claude(path)
        from fill import get_estimator
        try:
            estimator = None if fill == "none" else get_estimator(fill)
        except ValueError as e:  # unknown fill spec
            raise HTTPException(400, str(e))
        if ocr or estimator:
            photo = Image.open(path).convert("RGB")
            with ThreadPoolExecutor(max_workers=1) as pool:  # OCR (Apple Vision) runs alongside fill
                ocr_job = pool.submit(add_label_text, photo, bottles) if ocr else None
                if estimator:
                    try:
                        add_fill(photo, bottles, estimator)
                    except ValueError as e:
                        raise HTTPException(400, str(e))
                if ocr_job:
                    ocr_job.result()
    return PhotoOut(image=name, pipeline=pipeline, bottles=bottles)
