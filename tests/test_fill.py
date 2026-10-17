"""Fill geometry on a drawn bottle: a 30 px neck, a shoulder widening to 100 px, a 100 px body, dark liquid below
row 270. No model runs: the outline is the drawn mask and the liquid line comes from a stub or the edge finder."""
import numpy as np
import pytest
from PIL import Image

from fill import EdgeLine, Geometry, UnknownEstimator, get_estimator

LINE = 270
ITEM = {"brand": "Test", "product": "Bottle", "size": "700", "unit": "ml", "liquid": "rum"}


def bottle(liquid=(70, 35, 10), air=(225, 225, 225)) -> tuple[Image.Image, np.ndarray]:
    mask = np.zeros((400, 200), bool)
    for y in range(40, 380):
        half = 15 if y < 120 else 50 if y >= 160 else 15 + (y - 120) * 35 // 40
        mask[y, 100 - half:100 + half] = True
    img = np.full((400, 200, 3), 128, np.uint8)
    img[mask] = air
    img[LINE:][mask[LINE:]] = liquid
    return Image.fromarray(img), mask


class Stub:
    name, uses_torch, uses_api = "stub", False, False

    def __init__(self, y):
        self.y = y

    def find(self, crop, mask, full_y, base_y, item):
        return self.y


def test_volume_of_revolution():
    crop, mask = bottle()
    f = Geometry(Stub(LINE)).estimate(crop, [50, 40, 150, 380], ITEM, mask)
    assert f.visible
    full_y, base_y = f.debug["full_y"], f.debug["base_y"]
    assert 110 <= full_y <= 130  # where the neck widens into the shoulder
    assert base_y == 379  # flat base seen side on
    width = mask.sum(axis=1).astype(float)
    expected = (width[LINE:base_y + 1] ** 2).sum() / (width[full_y:base_y + 1] ** 2).sum() * 700
    assert f.fill_ml == pytest.approx(expected, abs=0.1)
    assert 300 < f.fill_ml < 345  # body below the line over body + shoulder, about 46%


def test_edge_finder_finds_the_line():
    crop, mask = bottle()
    f = Geometry(EdgeLine()).estimate(crop, [50, 40, 150, 380], ITEM, mask)
    assert abs(f.debug["line_y"] - LINE) <= 3


def test_no_line_and_no_contrast_give_null():
    crop, mask = bottle()
    assert Geometry(Stub(None)).estimate(crop, [], ITEM, mask).fill_ml is None
    flat, mask = bottle(liquid=(225, 225, 225))
    f = Geometry(Stub(LINE)).estimate(flat, [], ITEM, mask)
    assert f.fill_ml is None and f.note.startswith("unsure")


def test_tiny_outline_is_not_measured():
    mask = np.zeros((50, 50), bool)
    mask[10:20, 20:30] = True
    f = Geometry(Stub(15)).estimate(Image.new("RGB", (50, 50)), [], ITEM, mask)
    assert not f.visible and f.note == "bottle outline too small to measure"


def test_get_estimator():
    assert not get_estimator("geometry:edge").uses_api
    assert get_estimator("geometry:opus-grid").uses_api
    assert get_estimator("vlm:opus").uses_api
    for bad in ("bogus", "vlm:gpt-4", "geometry:nonsense"):
        with pytest.raises(UnknownEstimator):
            get_estimator(bad)


def test_unknown_size_gives_null_fill():
    crop, mask = bottle()
    for estimator in (Geometry(Stub(LINE)), get_estimator("vlm:opus")):  # returns before any model call
        f = estimator.estimate(crop, [50, 40, 150, 380], ITEM | {"size": "", "unit": ""}, mask)
        assert f.fill_ml is None and f.note == "bottle size unknown"
