"""HTTP API: one photo in, the shared output schema out. Web UI at /.

Usage (from the repo root):
    .venv/bin/uvicorn api:app --app-dir src
    curl -F image=@photo.jpg 'localhost:8000/identify?pipeline=local&fill=geometry:opus-grid'

Pipelines:
  local   every bottle SAM 3 finds (src/find.py). Bottles are not identified: brand and product are null and the
          size is assumed to be a 700 ml spirit.
  claude  Claude reads every label (src/identify.py, the baseline).
Fill runs on alcohol bottles (every bottle in `local`): the crop check (qc.GeometryQC) first, then the estimator
named by `fill` (src/fill.py); `none` skips it. Upside-down (optic) bottles are not measured.
"""
import io
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
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, PrivateAttr

import data_routes
import identify
import judge_routes
import testing_routes
from boxes import padded_crop
from photos import open_photo
from schema import ALCOHOL
from vlm import ERRORS as VLM_ERRORS
from vlm import Refusal, has_api_key

WEB = Path(__file__).resolve().parent.parent / "web"
PREVIEW_SIDE = 2400
FILL_WORKERS = 16  # concurrent Claude calls per photo
ASSUMED_SIZE_ML = 700  # unidentified bottles: a UK standard spirit bottle, so fill_ml can be given

app = FastAPI(title="bottle-vision")
app.include_router(testing_routes.router)  # Testing tab: /testing and /api/test (read-only)
app.include_router(data_routes.router)  # Data tab: /data and /api/data (read-only)
app.include_router(judge_routes.router)  # Judge tab: /judge and /api/judge (one append-only POST)
_torch = threading.Lock()  # one local model on the GPU at a time


class BottleOut(identify.Bottle):
    brand: str | None  # null until identified
    product: str | None
    liquid: str | None
    glass: Literal["clear", "green", "amber", "frosted", "opaque"] | None  # SAM 3 does not read the glass
    id_source: Literal["label", "shape", "none"]
    fill_note: str | None = None  # why fill is null: crop rejected, not alcohol, line not visible
    label_text: str | None = None  # what Apple Vision OCR reads on the label (macOS only); not used for the ID
    _found: object = PrivateAttr(default=None)  # find.Found, so fill reuses the whole-photo outline


class PhotoOut(BaseModel):
    image: str
    pipeline: Literal["local", "claude"]
    bottles: list[BottleOut]


def identify_all(photo: Image.Image) -> list[BottleOut]:
    """Every bottle SAM 3 finds, in reading order, none of them identified."""
    from find import find_bottles

    with _torch:
        found = find_bottles(photo)
    bottles = []
    for pos, f in enumerate(found, 1):
        b = BottleOut(position=pos, description=f"not identified; size assumed {ASSUMED_SIZE_ML} ml", bbox=f.box,
                      brand=None, product=None, category="spirit", liquid=None, size=ASSUMED_SIZE_ML, unit="ml",
                      glass=None, fill_ml=None, id_source="none", fill_visible=False,
                      fill_note="upside down (optic): not measured" if f.upside_down else None)
        b._found = f
        bottles.append(b)
    return bottles


@cache
def claude_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def identify_claude(path: Path) -> list[BottleOut]:
    try:
        bottles = identify.identify(claude_client(), path)
    except Refusal as e:
        raise HTTPException(422, str(e)) from e
    except anthropic.APIError as e:
        raise HTTPException(502, f"Claude API: {e}") from e
    return [BottleOut(**b.model_dump()) for b in bottles]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/demo", include_in_schema=False)
def demo_page() -> FileResponse:
    """The web UI in demo mode: dropping the demo photo replays web/demo/result.json (see src/make_demo.py)."""
    return FileResponse(WEB / "index.html")


@app.get("/demo/{name}", include_in_schema=False)
def demo_file(name: str) -> FileResponse:
    path = WEB / "demo" / Path(name).name
    if not path.is_file():
        raise HTTPException(404, f"no demo file {name}")
    return FileResponse(path)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/preview")
def preview(image: UploadFile) -> Response:
    """The photo as a JPEG any browser can show (HEIC included), upright, plus the size the boxes refer to."""
    try:
        img = ImageOps.exif_transpose(Image.open(image.file)).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(400, f"{image.filename} is not an image") from e
    width, height = img.size
    img.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg", headers={"X-Width": str(width), "X-Height": str(height)})


def add_fill(photo: Image.Image, bottles: list[BottleOut], estimator) -> None:
    """Fill level for each alcohol bottle, in place: crop at full resolution, check the crop, estimate.

    Outlines and the crop check run one bottle at a time under the GPU lock. Each cleared bottle's estimate goes to
    a thread pool straight away, so the Claude calls run in parallel with each other and with the next outline."""
    from fill import segment
    from qc import GeometryQC

    def estimate(b: BottleOut, crop: Image.Image, box: list[float], item: dict, mask) -> None:
        with _torch if estimator.uses_torch else nullcontext():
            try:
                f = estimator.estimate(crop, box, item, mask)
            except VLM_ERRORS as e:  # a failed call costs one bottle's fill, not the photo
                b.fill_note = f"fill failed: {e}"
                return
        b.fill_ml, b.fill_visible = f.fill_ml, f.visible
        b.fill_note = None if f.visible else (f.note or "liquid line not visible")

    with ThreadPoolExecutor(max_workers=FILL_WORKERS) as pool:
        jobs = []
        for b in bottles:
            if b.category not in ALCOHOL:
                b.fill_note = "not alcohol"
                continue
            if b.fill_note:  # already ruled out (upside down)
                continue
            crop, box = padded_crop(photo, b.bbox)
            item = b.model_dump()
            with _torch:
                mask = b._found.outline(crop.size, box) if b._found else segment(crop, box)
                verdict = GeometryQC().check(crop, box, item, None, mask)
            if not verdict.ok:
                b.fill_note = f"crop rejected: {verdict.reason}"
                continue
            jobs.append(pool.submit(estimate, b, crop, box, item, mask))
        for job in jobs:
            job.result()


def add_label_text(photo: Image.Image, bottles: list[BottleOut]) -> None:
    """OCR on each bottle's full-resolution crop (src/ocr.py), in place. Skipped off macOS."""
    try:
        from ocr import read_text
    except ImportError:
        return
    with tempfile.TemporaryDirectory() as tmp:
        for b in bottles:
            crop_path = Path(tmp) / f"{b.position}.jpg"
            padded_crop(photo, b.bbox)[0].save(crop_path, quality=92)
            b.label_text = " | ".join(t for t, _ in read_text(crop_path)) or None


@app.post("/identify", response_model=PhotoOut)
def identify_photo(image: UploadFile, pipeline: Literal["local", "claude"] = "local",
                   fill: str = "geometry:opus-grid", ocr: bool = True) -> PhotoOut:
    """One photo of bottles in, every bottle out in reading order, with a fill level for alcohol bottles.
    pipeline=local with fill=none (or a local fill estimator) needs no API key."""
    from fill import UnknownEstimator, get_estimator

    try:
        estimator = None if fill == "none" else get_estimator(fill)
    except UnknownEstimator as e:
        raise HTTPException(400, str(e)) from e
    if (pipeline == "claude" or (estimator and estimator.uses_api)) and not has_api_key():
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set in .env: use pipeline=local with fill=none or "
                                 "fill=geometry:edge")
    name = Path(image.filename or "").name or "upload"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(image.file.read())
        try:
            photo = open_photo(path)
        except UnidentifiedImageError as e:
            raise HTTPException(400, f"{name} is not an image") from e
        bottles = identify_all(photo) if pipeline == "local" else identify_claude(path)
    if ocr or estimator:
        with ThreadPoolExecutor(max_workers=1) as pool:  # OCR (Apple Vision) runs alongside fill
            ocr_job = pool.submit(add_label_text, photo, bottles) if ocr else None
            if estimator:
                add_fill(photo, bottles, estimator)
            if ocr_job:
                ocr_job.result()
    return PhotoOut(image=name, pipeline=pipeline, bottles=bottles)
