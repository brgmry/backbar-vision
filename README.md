# bottle-vision

Camera-based stock tracking for restaurant bars. From a photo: what bottle, what size, what liquid, how much is left.

## Tasks
1. Identify bottle type, size and liquid.
2. Estimate fill level (% and ml).

## Data
- `data/raw/` original photos, untouched
- `data/labels.csv` one row per photo: ground truth for eval
- `data/labelled/` exported annotations (if we fine-tune)

## Setup
Keys go in `.env`, never committed.

## Run the baseline
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # add your key
.venv/bin/python src/identify.py data/raw
```
Writes `data/predictions.csv`: brand, product, type, size, liquid, glass, confidence per photo.
