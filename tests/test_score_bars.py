import score_bars
from score import score_photo


def test_parse_label():
    rows = score_bars.parse_label("Gordon's | London Dry Gin | 70cl\n\nJameson\n")
    assert [(r["brand"], r["product"], r["size"]) for r in rows] == [
        ("Gordon's", "London Dry Gin", "70cl"), ("Jameson", "", "")]


def test_labels_score_like_inventory():
    inventory = score_bars.parse_label("Gordon's | London Dry Gin | 70cl\nJameson | Irish Whiskey")
    preds = [{"brand": "Gordon's", "product": "London Dry Gin", "category": "spirit", "size": "700", "unit": "ml"}]
    s = score_photo(preds, inventory)
    assert s["recall"] == 1 / 2
    assert s["brand_recall"] == 1 / 2
    assert s["size_acc"] == 1 / 2


def test_load_labels_skips_empty_and_other_tiers(tmp_path, monkeypatch):
    monkeypatch.setattr(score_bars, "LABELS", tmp_path)
    (tmp_path / "bar_35.txt").write_text("Jameson | Irish Whiskey\n")
    (tmp_path / "bar_36.txt").write_text("\n")
    tiers = {r["file"]: r["tier"] for r in score_bars.read_csv(score_bars.BARS / "manifest.csv")}
    assert set(score_bars.load_labels()) == {"bar_35.jpg"}
    other = next(t for t in ("easy", "medium", "hard") if t != tiers["bar_35.jpg"])
    assert score_bars.load_labels(other) == {}
