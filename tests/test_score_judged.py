import csv
from collections import Counter

import score_judged
from score import wilson

COLS = ["card", "photo", "pair", "system", "text", "brand_given", "judge", "verdict", "note"]


def card(pair, system, verdict, judge="a", brand_given="yes"):
    return {"card": f"{pair}-{system}", "photo": "bar_01.jpg", "pair": pair, "system": system, "text": "x",
            "brand_given": brand_given, "judge": judge, "verdict": verdict, "note": ""}


def test_wilson():
    assert wilson(0, 0) == (None, None)
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi


def test_right_share_leaves_out_cant_tell():
    assert score_judged.right_share(Counter(right=3, wrong=1, cant_tell=6)).startswith("75%")
    assert score_judged.right_share(Counter(cant_tell=2)) == "n/a"


def test_main(tmp_path, monkeypatch, capsys):
    rows = [card("p1", "claude", "right"), card("p1", "ours", "wrong"),
            card("p2", "claude", "wrong", brand_given="no"), card("p2", "ours", "right"),
            card("p3", "claude", "right"), card("p3", "ours", "cant_tell"),
            card("p1", "claude", "right", judge="b"), card("p1", "ours", "right", judge="b")]
    path = tmp_path / "verdicts.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    monkeypatch.setattr(score_judged, "VERDICTS", path)
    score_judged.main()
    out = capsys.readouterr().out
    assert "bottles both named: 2  both right 0  only claude 1  only ours 1  neither 0" in out  # p3 is can't tell
    assert "2 name a brand  right of decided 100%" in out  # the "Gin"-style answer on p2 is left out
    assert "same verdict 1  differ: {}" in out and "differ: {'wrong->right': 1}" in out
