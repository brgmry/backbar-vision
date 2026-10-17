import pytest

from schema import size_ml


@pytest.mark.parametrize(("row", "ml"), [
    ({"size": "700", "unit": "ml"}, 700),
    ({"size": "70", "unit": "cl"}, 700),
    ({"size": 1.75, "unit": "l"}, 1750),
    ({"size": "1", "unit": "pint"}, 568),
    ({"size": "70cl"}, 700),
    ({"size": "0,5l"}, 500),
    ({"size": "1ltr"}, 1000),
    ({"size": ""}, None),
    ({"size": None, "unit": "ml"}, None),
])
def test_size_ml(row, ml):
    assert size_ml(row) == (pytest.approx(ml) if ml else None)
