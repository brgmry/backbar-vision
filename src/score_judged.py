"""Score the blind card verdicts on the bar test set: is each system's name right for the bottle?

A card is one system's answer (brand and product) for one bottle on bar_01 to bar_05, shown with the bottle's crop
and the top web images for the name, never with the system's name. data/test_bars/card_verdicts.csv has one row per
card and judge: card, photo, pair, system, text, brand_given (yes, no), judge, verdict (right, wrong, cant_tell), note.
`pair` is the bottle; it has one card per system that named it.

Prints per judge and system the share of cards judged right among those with a verdict (Wilson 95% interval), the
head-to-head on bottles both systems named, and, for two judges, how often they agree on the cards both judged.

Usage: python src/score_judged.py
"""
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from schema import read_csv
from score import wilson

VERDICTS = Path(__file__).resolve().parent.parent / "data/test_bars/card_verdicts.csv"
SYSTEMS = ("claude", "ours")


def right_share(counts: Counter) -> str:
    decided = counts["right"] + counts["wrong"]
    if not decided:
        return "n/a"
    lo, hi = wilson(counts["right"], decided)
    return f"{counts['right'] / decided:.0%} ({lo:.0%} to {hi:.0%})"


def per_system(rows: list[dict]) -> None:
    """All cards, then only those that name a brand: an answer like "Gin" with no brand is never right."""
    for system in SYSTEMS:
        mine = [r for r in rows if r["system"] == system]
        counts = Counter(r["verdict"] for r in mine)
        branded = Counter(r["verdict"] for r in mine if r["brand_given"] == "yes")
        print(f"  {system:7} {len(mine):4} cards  right {counts['right']:3}  wrong {counts['wrong']:3}  "
              f"can't tell {counts['cant_tell']:3}  right of decided {right_share(counts)}")
        print(f"  {'':7} {sum(branded.values()):4} name a brand  right of decided {right_share(branded)}")


def head_to_head(rows: list[dict]) -> None:
    """Bottles both systems named, where neither card is can't tell."""
    by_pair = defaultdict(dict)
    for r in rows:
        by_pair[r["pair"]][r["system"]] = r["verdict"]
    both = [v for v in by_pair.values() if set(v) == set(SYSTEMS) and "cant_tell" not in v.values()]
    outcome = Counter(("claude" if v["claude"] == "right" else "") + ("+ours" if v["ours"] == "right" else "") for v in both)
    print(f"  bottles both named: {len(both)}  both right {outcome['claude+ours']}  only claude {outcome['claude']}  "
          f"only ours {outcome['+ours']}  neither {outcome['']}")


def agreement(rows: list[dict]) -> None:
    by_card = defaultdict(dict)
    for r in rows:
        by_card[r["card"]][r["judge"]] = r
    judges = sorted({r["judge"] for r in rows})
    for a, b in combinations(judges, 2):
        print(f"{a} vs {b}")
        for system in SYSTEMS:
            shared = [v for v in by_card.values() if a in v and b in v and v[a]["system"] == system]
            same = sum(v[a]["verdict"] == v[b]["verdict"] for v in shared)
            flips = Counter(f"{v[a]['verdict']}->{v[b]['verdict']}" for v in shared if v[a]["verdict"] != v[b]["verdict"])
            print(f"  {system:7} {len(shared):4} shared cards  same verdict {same}  differ: {dict(flips)}")


def main() -> None:
    rows = read_csv(VERDICTS)
    for judge in sorted({r["judge"] for r in rows}):
        mine = [r for r in rows if r["judge"] == judge]
        print(judge)
        per_system(mine)
        head_to_head(mine)
    agreement(rows)


if __name__ == "__main__":
    main()
