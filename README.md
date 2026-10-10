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

## Run the API
```
uv pip install -p .venv -r requirements.txt
.venv/bin/uvicorn api:app --app-dir src     # from the repo root
curl -F image=@photo.jpg 'localhost:8000/identify?pipeline=local'
```
`POST /identify` takes one photo and returns `{image, pipeline, bottles: [...]}` in the shared output schema. `pipeline=local` (default) runs the local pipeline with k=3 reference crops per item from every photo in `data/crops`, so photos from `data/raw` score themselves; test with new photos. The first local call loads the models (about a minute), later calls take a few seconds. `pipeline=claude` runs the baseline and needs the key in `.env`. Interactive docs at `localhost:8000/docs`.

## Dashboard
`.venv/bin/python src/dashboard.py` then open http://localhost:8501 (read-only view of photos, boxes, predictions, scores and pipeline status).
