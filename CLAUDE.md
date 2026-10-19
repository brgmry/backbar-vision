# bottle-vision

Cameras in restaurant bars that track stock from photos. Started as a Saturday hack by Bergen and Allen; now being cleaned up and written up for open source. README.md has the results and how to run everything.

## What it does
From one photo, for each bottle in reading order (shelf rows top to bottom, then left to right):
1. Identify it: brand, product, size, liquid.
2. Estimate how much is left, in ml.

Objective (agreed 10 Oct): beat the Claude baseline (`src/identify.py`, open-ended, claude-opus-5-5) by 10+ points of item recall on the human-labelled bar test set, at zero API cost per photo.

Output per bottle: `{position, description, bbox, brand, product, category, liquid, size, unit, glass, fill_ml, id_source, fill_visible}`, plus `fill_note` and `label_text` from the API. `bbox` is `[x, y, w, h]` in original-photo pixels. `fill_ml` is always ml, and null when `fill_visible` is false. `id_source` is `label` (read off the bottle), `shape` (guessed) or `none` (not identified). Fill % is derived from `fill_ml` and `size`, never output.

## Layout
- `data/raw/` the kitchen photos. Never edit, rename or overwrite.
- `data/labelled/` ground truth: `inventory.csv` (the 12 items in every kitchen photo), `fill.csv` (measured fill levels), `crop_exclude.csv` (crops rejected by hand, scores the crop check).
- `data/pour/` the pour test; `data/test_bars/` the bar test set; `data/predictions_*.csv` every run; `data/results.csv` the leaderboard.
- `src/` the pipeline, scorer and app; `src/experiments/` the local identification experiments. Shared helpers live in `photos.py`, `boxes.py`, `models.py`, `schema.py`; use them instead of copying HEIC loading, box maths, devices or unit tables.
- `tests/` fast offline tests: `.venv/bin/python -m pytest`.

## Pipeline
- Find: `src/find.py`, SAM 3 with the text prompt "bottle" on the whole photo.
- Identify: `src/identify.py`, Claude reading the labels, is the baseline to beat. The local identification experiments (catalogue retrieval, trained head, OCR fusion, local VLMs) are in `data/results.csv`.
- Crop check: `src/qc.py`. Fill: `src/fill.py`, outline from SAM, liquid line from Claude picking a numbered tick, volume of revolution.
- `src/experiment.py` runs any fill estimator on any predictions CSV, so identification and fill swap independently. `src/api.py` serves the pipeline and the web UI.

## Identification design (agreed 10 Oct)
- Crop, then a backbone (DINOv2 or SigLIP2) with a trained projection head, then a brand classifier for brands with at least `MIN_PER_BRAND` training images (30 in src/experiments/train_head.py; the 10 Oct plan said 100) and nearest-neighbour retrieval against the catalogue for the exact product, or "unknown". OCR on each crop (Apple Vision) is fused with the image score and reported as its own leaderboard row.
- No distillation: Claude never labels training data. Training data comes from external sources, cleaned without a model where possible (for example OCR checks on in-the-wild brand photos).
- Unknown thresholds and hyperparameters are tuned on a validation split of the external data, never on a test set.

## Ground truth is the answer, never an input
The only input is photos. `data/labelled/`, `data/crops/labels.csv` and the bar test labels are read by the scorer and nothing else: not as a reference library, not as a stock list in a prompt, not to pick which catalogue images to download. Pretrained models and broad external catalogues are fine as long as nothing in them was chosen by looking at our answers. The `data/reference/` packshots were picked from the answer list, so they are out. The leave-one-photo-out matcher (97.6% recall) broke this rule; its row stays in `data/results.csv`, marked invalid.

## Eval protocol
- Kitchen set (`data/raw/`): 7 photos, 12 items, scored by `src/score.py`.
- Bar test set (`data/test_bars/`): 82 real bar photos from web search (bar_01 to bar_34 from DuckDuckGo, bar_35 to bar_82 from Wikimedia Commons with licence, author and an easy/medium/hard tier in `manifest.csv`). Labelled by people only, never by a model, since the baseline is a model. Never used for training: any training image within phash distance 6 of `data/test_bars/phashes.txt` is dropped.
- Fill: measured labels only (poured or weighed), never model-made. Report the error in points of bottle size and the share of bottles answered.
- Report item recall, brand recall and bar recall (spirits, liqueurs, wine, beer, mixers). Size, category and liquid only count when something actually predicts them.
- Thresholds and prompts are tuned on our own kitchen and pour photos, never on a test set.

## Rules
- Eval first: score every approach before calling it working.
- Fill % is volume, not height. Shaped bottles need the outline's width per row.
- Opaque bottles have no visible liquid line: `fill_visible` false and `fill_ml` null, never a silent guess.
- Secrets live in `.env` (gitignored). Never commit keys, weights (`*.pt`, `*.torch`), `runs/` or `vendor/`.
- Claude calls cost money. Prefer the answer cache in `runs/vlm_cache/` and single runs; keep prompts byte-identical when refactoring so cached answers still hit.
- Prefer the simplest thing that hits the eval bar.
- Small commits, pushed often. Two people work in this repo, so pull before you start.
