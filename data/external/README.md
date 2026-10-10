
## wild/ : in-the-wild bottle photos by brand (pre-training only)

- `brands.csv`: ~400 spirits/liqueur brands with source citations (Drinks International Millionaires' Club and Brands Report, The Spirits Business Brand Champions, IWSR, CGA UK on-trade lists). Compiled from those rankings, **not** from `data/labelled/`. Our ground truth was never opened to build it. Citations are list-level, compiled from memory of 2023-2025 editions, not scraped tables.
- `wild/<slug>/<n>.jpg`: up to 60 photos per brand for the 300 most-cited brands. Longest side <= 1024px, min side >= 200px. Collector: `src/collect_wild.py [max_brands] [per_brand]` (resumable, `_done` marker per brand; `merge` rebuilds the manifest).
- Sources in priority order: Wikimedia Commons (search API, free licences), Openverse (CC, anon limit 200 req/day so roughly the first 180 brands only), DuckDuckGo image search via `ddgs` (queries `<brand> bottle bar|cocktail|hand`, licence unknown). No Flickr key in env, so Flickr direct was skipped (Openverse returns Flickr CC images anyway).
- `wild/manifest.csv`: brand, slug, file, source, url, page, licence, w, h, phash.
- Dedup: pHash, Hamming <= 6 within a brand, <= 2 across brands. Images within distance 6 of `data/test_bars/phashes.txt` are removed (no train/test overlap).
- Labels are **weak**: brand = the search query. Many images show several brands on a shelf, some are packshots or watermarked stock. Use for contrastive pre-training or with a detector crop + filter, not as clean classification labels. Licence "unknown" (DDG) images are for internal research use only, never redistribute.

## winesensed/ : WineSensed (Vivino user photos of wine labels), all 24 shards

- Source: https://huggingface.co/datasets/christopher/winesensed (NeurIPS 2023 D&B, "Learning to Taste", Bender et al.). Licence **CC BY-NC-ND 4.0** (research only, no derivatives redistributed).
- On disk: `data/wines/train-000NN-of-00024.parquet`, all 24 shards, ~36 GB, kept as parquet (images as bytes). The first 8 shards hold 338,214 rows, 336,568 unique images and 205,360 `vintage_id`s; the full set is ~897k images. Columns: image (bytes), review, wine, year, wine_alcohol, country, region, price, rating, grape, vintage_id, vintage_page_url, winery_id.
- Class label for retrieval: `vintage_id` (or `winery_id` for coarser). Mostly one phone photo per vintage, so it is a long-tail, few-shot set: good for contrastive pre-training of label embeddings, not for classification.
- `one-shot-benchmark/`: github.com/EdenHuang/winesensed-one-shot-benchmark (image-free split files and hashes for a 1,000-class one-shot wine label retrieval benchmark on WineSensed; companion to arXiv 2404.08820). No licence file.
- Fetch: `set -a; . ./.env; set +a; hf download christopher/winesensed --repo-type dataset --local-dir data/external/winesensed --include "data/wines/*"`.

## beer-classification-ru/ : Russian retail beer bottles and cans

- Source: https://huggingface.co/datasets/ramnck/beer-classification. Licence **MIT**.
- 5,499 phone photos, 65 classes (brand + variant, e.g. `baltika_3`, `kozel_dark`, `hoegaarden_cherry`). Layout `images/{train,validation,test}/<class>/*.jpg`, 528 MB. Extracted from the HF parquet conversion.

## beerbb-1k/ : BeerBB-1K beer bottle localisation

- Source: https://github.com/devfoo-one/BeerBB-1K (Beuth Hochschule Berlin, 2017). **No licence stated**: internal research only.
- 1,000 smartphone photos (1000x750), 1,180 bottle boxes, 35 beer brands, plus `isOpen` flag. Layout `BeerBB-1K/NNNN.jpg` + `NNNN.json` (`[{x,y,w,h,brand,isOpen}]`). 90 MB.
- Useful for detector fine-tuning and brand crops in cluttered real scenes.

## wines-label-dataset/ : wine label boxes (Pascal VOC)

- Source: https://github.com/qanastek/WinesLabelDataset. **No licence stated**.
- 254 annotated images (single class `label`, box around the wine label) in `dataset_annotated/` (`(n).jpg` + `(n).xml`), plus unannotated images in `non_annotated/`. 19 MB. Label localisation only, no identity.

## imagenet-bottles/ : ImageNet-1k `wine bottle` and `beer bottle` classes

- Source: https://huggingface.co/datasets/mlnomad/imnet1k_wine_bottle and `.../imnet1k_beer_bottle` (ImageNet-1k subsets). ImageNet terms: **non-commercial research only**.
- 2,700 images (1,350 per class; train + validation 50 each). Layout `<class>/{train,validation}/NNNNN.jpg`. 312 MB. Generic in-the-wild bottles, no brand labels: negatives, detector or domain pre-training only.

## whiskey-classification/ : Scotch whisky bottle photos (YOLO split, 20 classes)

- Source: https://huggingface.co/datasets/spacenship/whiskeyClassification (DGIST summer intern project, 2025). **No licence or card**: internal research only.
- Repo is 81 GB of PNG. We take only `dataset/` (the YOLO-style split): 20 classes, 350 train / 100 val / 50 test images per class, 10,000 images with YOLO `labels/<split>/*.txt`, plus `labeling.py`. Skipped: the `labels/<name>/` raw-frame tree (12 classes, ~26k PNG, 58 GB), which duplicates the same videos.
- PNGs (22.6 GB) are converted in place to JPEG q95 (`src/external/png_to_jpg.py`) to stay under ~20 GB.
- **Class mapping** `classes.csv`: file prefix `N_` is `whiskey_list.index(name) + 1` from the repo's `makeDataset.py`. I verified it for 12 of 20 classes by matching LFS sha256 of `dataset/images` files against the named `labels/<name>/` tree (all 500 images per class matched). The other 8 (ardbeg 5, ardbeg 10, balvenie 12/14, glendronach 12/15, glenfiddich 12/15) rely on the script only.
- `makeDataset.py` keeps every 5th of each class's sequentially numbered captures, so near-duplicates within a class are common: split by class or video, not by frame, for any eval.
- Fetch: `set -a; . ./.env; set +a; hf download spacenship/whiskeyClassification --repo-type dataset --local-dir data/external/whiskey-classification --include "dataset/*" --include "labeling.py"`, then `python src/external/png_to_jpg.py data/external/whiskey-classification/dataset`.
- Brands chosen by the dataset author, not by us. Check overlap with our inventory only when scoring, never filter on it.

## ttb-cola-spirits/ : US TTB COLA public registry label images (spirits)

- Source: https://ttbonline.gov/colasonline/publicSearchColasBasic.do (US Treasury TTB Certificate of Label Approval registry). US government public record; label artwork copyright stays with the brand owner, so internal research only.
- Method (`src/external/ttb_cola.py`, resumable): basic search, CSV export (`publicSaveSearchResultsToFile.do`, capped at 1,000 rows per query) over 1-week windows every 4 weeks from Jan 2024, class/type ranges 100-699 (whisky, gin, vodka, rum, brandy, liqueurs, specialties) and 977-983 (tequila, mezcal, agave, sake). 4,003 COLAs listed, 3,000 sampled at random (seed 0) into `index.csv`. Per COLA: open `viewColaDetails.do?action=publicFormDisplay&ttbid=...` then fetch every `publicViewAttachment.do?filename=...&filetype=l` image in the same session.
- Layout: `index.csv` (TTB ID, permit, serial, completed date, fanciful name, brand name, origin, class/type code + description), `images.csv` (ttb_id, file, original filename, which often says FRONT/BACK/NECK), `images/<ttbid>_<k>.<ext>`.
- These are flat label artwork (front, back, neck), not photos of bottles. Good for label-text/logo embedding pre-training and synthetic compositing onto bottle renders.
- Rate: the server is slow (~80 KB/s per connection), so ~10 COLAs/min with 8 workers and 0.5 s sleeps. It was still running at hand-off; rerun the same command to resume: `.venv/bin/python src/external/ttb_cola.py 3000`.
