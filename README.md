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
