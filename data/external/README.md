
## wild/ : in-the-wild bottle photos by brand (pre-training only)

- `brands.csv`: ~400 spirits/liqueur brands with source citations (Drinks International Millionaires' Club and Brands Report, The Spirits Business Brand Champions, IWSR, CGA UK on-trade lists). Compiled from those rankings, **not** from `data/labelled/`. Our ground truth was never opened to build it. Citations are list-level, compiled from memory of 2023-2025 editions, not scraped tables.
- `wild/<slug>/<n>.jpg`: up to 60 photos per brand for the 300 most-cited brands. Longest side <= 1024px, min side >= 200px. Collector: `src/collect_wild.py [max_brands] [per_brand]` (resumable, `_done` marker per brand; `merge` rebuilds the manifest).
- Sources in priority order: Wikimedia Commons (search API, free licences), Openverse (CC, anon limit 200 req/day so roughly the first 180 brands only), DuckDuckGo image search via `ddgs` (queries `<brand> bottle bar|cocktail|hand`, licence unknown). No Flickr key in env, so Flickr direct was skipped (Openverse returns Flickr CC images anyway).
- `wild/manifest.csv`: brand, slug, file, source, url, page, licence, w, h, phash.
- Dedup: pHash, Hamming <= 6 within a brand, <= 2 across brands. Images within distance 6 of `data/test_bars/phashes.txt` are removed (no train/test overlap).
- Labels are **weak**: brand = the search query. Many images show several brands on a shelf, some are packshots or watermarked stock. Use for contrastive pre-training or with a detector crop + filter, not as clean classification labels. Licence "unknown" (DDG) images are for internal research use only, never redistribute.

## winesensed/ : WineSensed (Vivino user photos of wine labels), all 24 shards

- Source: https://huggingface.co/datasets/christopher/winesensed (NeurIPS 2023 D&B, "Learning to Taste", Bender et al.). Licence **CC BY-NC-ND 4.0** (research only, no derivatives redistributed).
- On disk: `data/wines/train-000NN-of-00024.parquet`, all 24 shards, 33 GB, kept as parquet (images as bytes). 1,014,630 rows (one per review photo) across 421,672 `vintage_id`s. Columns: image (bytes), review, wine, year, wine_alcohol, country, region, price, rating, grape, vintage_id, vintage_page_url, winery_id.
- Class label for retrieval: `vintage_id`. About 2.4 phone photos per vintage on average, so it is a long-tail, few-shot set: good for contrastive pre-training of label embeddings, not for classification.
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
- PNGs (22.6 GB) converted in place to JPEG q95 (`src/external/png_to_jpg.py`): 10,000 images, 3.9 GB on disk. Layout `dataset/images/{train,val,test}/<classid>_NNNN.jpg`, `dataset/labels/<split>/<same>.txt`.
- The YOLO `.txt` files are **not identity labels**: each line is `39 0 cx cy w h` (COCO class 39 = bottle, apparently detector output), one box per bottle. Identity comes only from the filename prefix via `classes.csv`. Use the boxes to crop.
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

## retail_<source>/ : retailer product catalogues (pre-training only)

Collector: `src/collect_retail.py <saq|systembolaget|drinkfinder> [--minutes N]`. Whole catalogues in seeded random order, never filtered by `data/labelled/`. Resume-safe (skips product_ids already in `products.csv`; re-run continues). Max 1 request/second per host, 3 worker threads, UA `bottle-vision-hackathon/0.1 (you@example.com)`. Logs in `retail_<source>.log`.

Layout per source: `img/<product_id>/<n>.jpg` (RGB JPEG, transparent backgrounds flattened to white, n = gallery position) and `products.csv` (source, product_id, brand, name, category, volume_ml, abv, url, image_urls; image_urls space-separated). Images are clean studio packshots, mostly front-only: use as a domain-gap complement to bar photos, not a substitute.

### retail_saq (SAQ, Quebec)
- Source: public Magento storefront GraphQL `https://www.saq.com/graphql` (`products(search:"")`, 100 per page, header `Store: en`). ~30,000 products incl. delisted. Images `www.saq.com/media/catalog/product/...` at native res (~1000x1500).
- Terms: robots.txt does not disallow /graphql or /media/catalog. Retailer imagery, copyright SAQ/producers: internal research only, never redistribute.
- Counts (first 29 min run): 1,316 products, 1,401 images (94% one image, 80 with 2-5), 130 MB. brand, volume_ml, abv filled for all (from `nom_producteur`, `meta_description` "Size :", `pourcentage_alcool_par_volume`). Category = `identite_produit` (English). Wine-heavy (~80%) as the catalogue is.

### retail_systembolaget (Systembolaget, Sweden)
- Source: product sitemaps `https://www.systembolaget.se/sitemap-produkter-{ol,vin,sprit,cider-blanddrycker,alkoholfritt}.xml` (26,866 products), metadata from each product page's embedded JSON, image `product-cdn.systembolaget.se/productimages/<productId>/<productId>_800.png`. product_id = Systembolaget article number (URL), image id differs.
- Terms: robots.txt allows /produkt/. Public open data was withdrawn; their API needs a subscription key, not used. Internal research only.
- Counts: 1,102 products, 1,102 images (one each), 381 MB (800px, heavier JPEGs). Category is Swedish (`categoryLevel2`: Rött vin, Ale, Whisky...). Includes beer, cider, alcohol-free.

### retail_drinkfinder (Drink Finder, UK, Shopify)
- Source: `https://www.drinkfinder.co.uk/products.json?limit=250&page=N` (5,589 products), images from cdn.shopify.com at `?width=1000`.
- Terms: Shopify robots.txt does not disallow products.json. Internal research only.
- Counts: 1,720 products, 1,720 images (store has one image per product), 123 MB. volume_ml parsed from title (98%), abv rarely present (6%). Category = Shopify product_type (Red Wine, Single Malt Whisky, Rum, Beer, Liqueurs, Gin...).

### Skipped
The Whisky Exchange and Total Wine (403 to a polite UA), Master of Malt (429), Majestic and Drinks&Co (403). LCBO api.lcbo.dev returned INTERNAL_SERVER_ERROR on every product query after the first. Waitrose Cellar, Ocado, Tesco, Sainsbury's not attempted (time).

## rp2k/ : RP2K retail product dataset (Pinlan, NeurIPS-era, arXiv 2006.12634)

- Source: original Huawei OBS zip `https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/rp2k_dataset.zip` (6.33 GB, still live Oct 2026; pinlandata.com page returns 429). Paper: https://arxiv.org/abs/2006.12634.
- Licence: research use, as stated by the authors on pinlandata.com; no formal open licence shipped in the zip. Do not redistribute.
- Images: 384,311 (train 344,854, test 39,457). Classes: 2,388 SKUs (same folder names in train and test). Shelf crops from Chinese supermarkets, one product per image. 6.6 GB extracted; zip deleted.
- Layout: `all/{train,test}/<SKU name in Chinese>/<n>.jpg` (a few png/jpeg).
- `alcohol_classes.csv`: keyword match on class names (啤酒, 白酒, 黄酒, 酒 and so on, cooking wine, vinegar and cigarettes excluded). 265 classes, 40,209 images: beer 192 classes / 34,866 images, baijiu 43 / 3,498, rice wine 11 / 1,331, other alcohol 19 / 514. No western spirits (no whisky, vodka, gin, rum). Heuristic, not an official category map.
- The HF repo `mteb/rp2k` is only the 39,457-image test split as a retrieval benchmark, no class names. Not downloaded.

## beverages-stperez/ : branded beverage bottles and cans

- Source: https://huggingface.co/datasets/stperez/beverages. Licence: MIT (dataset card).
- Images: 4,428 (train 2,271, test 2,157), mostly png, a few webp/gif. Classes: 29 brands (beer: Budweiser, Corona, Stella Artois, Modelo, Michelob Ultra, Castle Lite, Balashi; water: Evian, Perrier, San Pellegrino, Nestle Pure Life; soft drinks and teas: Coca-Cola, Fanta, Sprite, Pepsi, Mountain Dew, Gatorade, Monster, Tropicana and others). 2.7 GB.
- Layout: `<Category>/<Company>/<brand>/{train,test}/<file>`; Balashi is `Beer/Balashi Brewery/{train,test}/` with no brand level. One stray `pepsi119.zip` in a test folder.
- Mostly web packshots and product photos.

## wine-images-126k/ : wine bottle packshots

- Source: https://huggingface.co/datasets/cipher982/wine-images-126k. Licence: CC BY 4.0 (`LICENSE`). Scraped from wine retailer sites.
- Images: 107,821 in 24 parquet shards `data/train-*.parquet`, columns `image_id, image (bytes), wine_name`. 5.8 GB. The duplicate tar `shards/` folder was deleted.
- Classes: every `wine_name` is unique, so one image per wine: an instance-retrieval / contrastive set, not a multi-shot classifier set. Companion text (producer, region, price) at `cipher982/wine-text-126k`, keyed by `image_id`, not downloaded.

## openimages-bottles/ : Open Images V7 images with Bottle, Wine or Beer boxes

- Source: Open Images (https://storage.googleapis.com/openimages/web/index.html). Box CSVs from `storage.googleapis.com/openimages/v5|v6/*-annotations-bbox.csv`, images from the public bucket `open-images-dataset.s3.amazonaws.com/<split>/<id>.jpg`. Licence: annotations CC BY 4.0, images listed by Google as CC BY 2.0 (Flickr, per-image attribution).
- Classes: 3 box labels, `Bottle` (/m/04dr76w), `Wine` (/m/081qc), `Beer` (/m/01599). Every image with at least one such box in validation, test and train.
- Layout: `images/{validation,test,train}/<ImageID>.jpg`; `boxes.csv` = split, label, plus the original Open Images columns (normalised XMin/XMax/YMin/YMax, IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside), only for images on disk. `meta/bb-*.csv` are the filtered source CSVs, `meta/fetch.py` re-runs the resumable download.
- Use: generic bottle detection/crops and negatives (c). No brand labels.
