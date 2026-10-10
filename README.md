# bottle-vision

Camera-based stock tracking for restaurant bars. From a photo: what bottle, what size, what liquid, how much is left.

## Tasks
1. Identify bottle type, size and liquid.
2. Estimate fill level (% and ml).

## Data
- `data/raw/` original photos, untouched
- `data/labelled/inventory.csv` ground truth: the 12 items in the set


## Setup
Keys go in `.env`, never committed.

## Run the baseline
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # add your key
.venv/bin/python src/identify.py data/raw
```
Writes `data/predictions.csv`: one row per bottle (run, photo, position left to right, description, bbox, brand, product, category, liquid, size, unit, glass, fill_ml, id_source, fill_visible).
