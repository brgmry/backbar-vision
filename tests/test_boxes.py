import numpy as np
import pytest
from PIL import Image

from boxes import iou, iou_xyxy, largest_blob, overlap_xyxy, padded_crop, resized_scale


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0
    assert iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)
    assert iou_xyxy([0, 0, 0, 0], [0, 0, 0, 0]) == 0  # empty boxes, no division by zero


def test_overlap_is_one_for_a_box_inside_another():
    assert overlap_xyxy([0, 0, 100, 100], [10, 10, 20, 20]) == 1
    assert overlap_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(0.5)


def test_padded_crop_stops_at_the_photo_edge():
    photo = Image.new("RGB", (100, 80))
    crop, box = padded_crop(photo, [0, 0, 20, 40])
    assert crop.size == (23, 46)  # 15% pad on the right and bottom only
    assert box == [0, 0, 20, 40]
    crop, box = padded_crop(photo, [50, 20, 20, 40])
    assert crop.size == (26, 52)
    assert box == [3, 6, 23, 46]


def test_largest_blob():
    mask = np.zeros((10, 10), bool)
    mask[0:2, 0:2] = True
    mask[4:9, 4:9] = True
    blob = largest_blob(mask)
    assert blob.sum() == 25 and not blob[0, 0]


@pytest.mark.parametrize(("size", "scale"), [((4032, 3024), 1.68), ((3024, 4032), 1.68), ((1000, 800), 1.0)])
def test_resized_scale(size, scale):
    assert resized_scale(size, 2400) == pytest.approx(scale)
