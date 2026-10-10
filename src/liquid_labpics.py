"""Liquid line from a SAM 2 net fine-tuned on LabPics to segment vessel / filled region / transparent region.

Model: sagieppel/Segment_Anything2_Trained_for_Semantic_Seg_Vessell_Fill_level (Apache-2.0), cloned to
vendor/sam2_vessel, weights from the author's Google Drive in vendor/weights (model_large.torch is a full
SAM2Base state dict, so no base SAM checkpoint is needed).

Setup (vendor/ is gitignored):
  git clone https://github.com/sagieppel/Segment_Anything2_Trained_for_Semantic_Seg_Vessell_Fill_level vendor/sam2_vessel
  download model_large.torch from https://drive.google.com/drive/folders/1j9Ovaq-hGeU_UzJv4RN2jpvYz6wxK3uB
  (or https://icedrive.net/s/CCkR6wRWuhg4fvX9zVfNC8u74A9N) into vendor/weights/

The vendored package is never installed: sam2/__init__.py pulls in hydra and setup.py builds a CUDA extension.
We register an empty `sam2` package pointing at the vendored folder, build the model from the yaml with a small
_target_ instantiator instead of hydra, and run it on mps/cpu in float32. Weights load with weights_only=True.

The net takes one dummy point and returns three masks for the whole image: 0 vessel, 1 filled, 2 transparent.
"""
import importlib
import sys
import types
from functools import cache
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy import ndimage

VENDOR = Path(__file__).resolve().parent.parent / "vendor"
REPO = VENDOR / "sam2_vessel"
WEIGHTS = {"large": (VENDOR / "weights/model_large.torch", "sam2_hiera_l.yaml"),
           "small": (VENDOR / "weights/model_small.torch", "sam2_hiera_s.yaml")}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RES = 1024  # the net sees a 1024x1024 canvas; training padded smaller images with black at bottom/right
MIN_LIQUID = 0.03  # share of the bottle mask below which there is no liquid to speak of
WALL = 0.15  # share of the bottle's width on each side used to find where the surface meets the walls
WALL_INSET = 0.05  # skip this share of the width at each edge: the glass itself
WALL_HIT = 0.25  # share of a row's wall band that must be liquid


def _instantiate(cfg):
    if isinstance(cfg, dict):
        args = {k: _instantiate(v) for k, v in cfg.items() if k != "_target_"}
        if "_target_" not in cfg:
            return args
        module, _, name = cfg["_target_"].rpartition(".")
        return getattr(importlib.import_module(module), name)(**args)
    if isinstance(cfg, list):
        return [_instantiate(v) for v in cfg]
    if isinstance(cfg, str):  # PyYAML (YAML 1.1) reads 1e-6 as a string; OmegaConf would give a float
        try:
            return float(cfg)
        except ValueError:
            return cfg
    return cfg


@cache
def predictor(size: str = "large"):
    if "sam2" not in sys.modules:  # skip the vendored __init__ (hydra); submodules import from the folder
        pkg = types.ModuleType("sam2")
        pkg.__path__ = [str(REPO / "sam2")]
        sys.modules["sam2"] = pkg
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    weights, cfg_name = WEIGHTS[size]
    cfg = yaml.safe_load((REPO / "sam2_configs" / cfg_name).read_text())["model"]
    model = _instantiate(cfg)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return SAM2ImagePredictor(model.to(DEVICE).eval())


def _canvas(crop: Image.Image, upscale: bool) -> tuple[np.ndarray, float]:
    r = RES / max(crop.size) if upscale else min(1.0, RES / max(crop.size))
    img = crop.convert("RGB").resize((round(crop.width * r), round(crop.height * r)), Image.BILINEAR)
    canvas = np.zeros((RES, RES, 3), dtype=np.uint8)
    canvas[:img.height, :img.width] = np.asarray(img)
    return canvas, r


def segment_classes(crop: Image.Image, size: str = "large", upscale: bool = False) -> np.ndarray:
    """(3, H, W) bool masks in crop pixels: vessel, filled (liquid), transparent."""
    pred = predictor(size)
    canvas, r = _canvas(crop, upscale)
    with torch.no_grad():
        pred.set_image_batch([canvas])
        _, coords, labels, _ = pred._prep_prompts(np.ones([1, 1, 2]), np.ones([1, 1]), box=None, mask_logits=None,
                                                  normalize_coords=True)
        sparse, dense = pred.model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        high_res = [f[-1].unsqueeze(0) for f in pred._features["high_res_feats"]]
        logits, _, _, _ = pred.model.sam_mask_decoder(
            image_embeddings=pred._features["image_embed"], image_pe=pred.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=True,
            repeat_image=False, high_res_features=high_res)
        masks = pred._transforms.postprocess_masks(logits, pred._orig_hw[-1])[0].cpu().numpy() > 0
    h, w = round(crop.height * r), round(crop.width * r)
    masks = masks[:, :h, :w]
    if r != 1:
        masks = np.stack([np.asarray(Image.fromarray(m).resize(crop.size, Image.NEAREST)) for m in masks])
    return masks


def wall_level(liquid: np.ndarray, bottle: np.ndarray, full_y: int = 0) -> int | None:
    """Row where the liquid meets the bottle's left and right walls: the middle of the surface ellipse.

    On each side, a band WALL of the row's width wide, starting WALL_INSET in from the mask edge (the bottle mask
    includes the glass wall, the liquid mask doesn't). The level on a side is the topmost row at or below full_y
    whose band is at least WALL_HIT liquid. Average left and right."""
    left = bottle.argmax(axis=1)
    right = bottle.shape[1] - 1 - bottle[:, ::-1].argmax(axis=1)
    width = right - left + 1
    sides = []
    for side in ("left", "right"):
        for y in np.flatnonzero(bottle.any(axis=1)):
            if y < full_y:
                continue
            inset, band = int(WALL_INSET * width[y]), max(1, round(WALL * width[y]))
            a = left[y] + inset if side == "left" else right[y] - inset - band + 1
            if liquid[y, a:a + band].mean() >= WALL_HIT:
                sides.append(y)
                break
    return round(np.mean(sides)) if sides else None


class LabPicsLine:
    name = "labpics"

    def __init__(self, size: str = "large", upscale: bool = False):  # upscaling to 1024 hurt on JW IMG_6345
        self.size, self.upscale = size, upscale
        self.debug: dict = {}

    def liquid(self, crop: Image.Image, mask: np.ndarray) -> np.ndarray:
        liquid = segment_classes(crop, self.size, self.upscale)[1] & mask
        labels, n = ndimage.label(liquid)
        if n > 1:  # the liquid is one body; drop specks (reflections, labels)
            liquid = labels == np.argmax(np.bincount(labels.ravel())[1:]) + 1
        return liquid

    def find(self, crop, mask, full_y, bottom_y, item):
        liquid = self.liquid(crop, mask)
        self.debug = {"liquid": liquid}
        if liquid.sum() < MIN_LIQUID * mask.sum():
            return None
        return wall_level(liquid, mask, full_y)
