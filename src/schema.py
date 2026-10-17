"""Categories and sizes shared by the pipeline, the scorer and the API."""
import re

ALCOHOL = {"spirit", "liqueur", "wine", "beer"}
BAR = ALCOHOL | {"mixer"}  # what a bar camera is for
TO_ML = {"ml": 1, "cl": 10, "l": 1000, "ltr": 1000, "litre": 1000, "liter": 1000,
         "g": 1, "kg": 1000,  # weight counted as ml: close enough for sauces
         "oz": 29.57,  # US fluid ounce
         "pint": 568}  # UK pint: our kitchen set is British
SIZE_TEXT = re.compile(r"([\d.,]+)\s*(ml|cl|l|ltr|litre|liter|g|kg|oz|pint)?")


def size_ml(row: dict) -> float | None:
    """Size in ml from `size` + `unit`, or from free text such as "70cl", "0,5l" or "1ltr". None if unreadable."""
    try:
        return float(row["size"]) * TO_ML.get(row.get("unit", ""), 1)
    except (TypeError, ValueError, KeyError):
        pass
    m = SIZE_TEXT.search(str(row.get("size", "")).lower())
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ".")) * TO_ML.get(m.group(2) or "ml", 1)
    except ValueError:
        return None
