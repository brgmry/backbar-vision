from score import match_photo, score_photo, size_ok

INVENTORY = [
    {"id": "1", "brand": "Johnnie Walker", "product": "Black Label 12 Year Old", "category": "spirit",
     "size": "700", "unit": "ml"},
    {"id": "2", "brand": "Bacardi", "product": "Carta Blanca Superior White Rum", "category": "spirit",
     "size": "700", "unit": "ml"},
    {"id": "3", "brand": "Heinz", "product": "Tomato Ketchup", "category": "sauce", "size": "460", "unit": "g"},
]


def pred(brand, product, category="spirit", size="70", unit="cl"):
    return {"brand": brand, "product": product, "category": category, "size": size, "unit": unit}


def test_each_item_matches_at_most_one_prediction():
    preds = [pred("Johnnie Walker", "Black Label"), pred("Johnnie Walker", "Black Label 12"), pred("Acme", "Gin")]
    matched = match_photo(preds, INVENTORY)
    assert sum(m is INVENTORY[0] for m in matched) == 1
    assert matched[2] is None


def test_score_photo():
    s = score_photo([pred("Johnnie Walker", "Black Label 12 Year Old"), pred("Acme", "Gin")], INVENTORY)
    assert s["recall"] == 1 / 3
    assert s["precision"] == 1 / 2
    assert s["count_error"] == -1
    assert s["bar_recall"] == 1 / 2  # the ketchup is not a bar item
    assert s["brand_recall"] == 1 / 3


def test_size_tolerance():
    assert size_ok(pred("", "", size="70", unit="cl"), INVENTORY[0])
    assert size_ok(pred("", "", size="750", unit="ml"), INVENTORY[0])  # within 10%
    assert not size_ok(pred("", "", size="1", unit="l"), INVENTORY[0])
