# Beverage datasets and image catalogues (10 Oct 2026)

Already in `research.md`, not repeated: Roboflow 481 (22 spirit brands), Roboflow Liquor-data, Roboflow Bottle fill level, WineSensed (897k Vivino label images, CC BY-NC-ND 4.0 on the HF mirror).

Roboflow pages 403 on every scripted fetch and there is no Roboflow key in `.env`, so every Roboflow row below comes from search snippets: **unverified**. Usefulness: 5 = drop straight into the pipeline, 1 = ignore.

## Datasets and catalogues

| Source | Images | Classes / SKUs | Labels | Licence | Use (1-5) | Notes |
|---|---|---|---|---|---|---|
| [Open Food Facts](https://world.openfoodfacts.org/data) ([API](https://openfoodfacts.github.io/openfoodfacts-server/api/)) | front packshot per product, bulk via AWS Open Data | world: 39.8k alcoholic beverages, 16.9k wines, 12.3k beers, 21.1k sodas, 10.3k syrups, 628 tonics. UK: 1,461 alcoholic, 150 spirits, 564 beers, 8.1k beverages | barcode, name, brand, quantity, category | DB ODbL, images CC BY-SA | **5** | Verified live. Jameson alone has 53 entries with front images. Best gallery for mixers, softs, syrups and mainstream spirits. Quantity field is noisy ("700 g"). Rate limits hard, add a UA and back off. |
| [Wine Images 126K](https://huggingface.co/datasets/cipher982/wine-images-126k) (HF) | 107,821 | ~107k wines (name per image) | wine name, links to [text set](https://huggingface.co/datasets/cipher982/wine-text-126k) (price, region, category) | CC BY 4.0 | **4** | Verified. Retailer packshots, 6 GB. Best open wine gallery with a commercial-friendly licence. Mirror: carolinembithe/wine-images-126k. |
| [LCBO.dev](https://lcbo.dev) GraphQL (`api.lcbo.dev/graphql`) | thumbnail per product | full LCBO catalogue (thousands) | `thumbnailUrl`, `upcNumber`, `unitVolumeMl`, `alcoholPercent`, category, producer | free, no auth; "non-commercial" per listings | **4** | Schema verified by introspection, but `products` query returned INTERNAL_SERVER_ERROR on 10 Oct. Size + UPC + image in one row is exactly the SKU record we need. |
| [TTB Public COLA Registry](https://www.ttb.gov/regulated-commodities/labeling/cola-public-registry) | label artwork for every US-approved label since 1999 | millions of approvals (beer, wine, spirits) | brand, class, TTB ID; images per approval | US government record | **3** | Verified page. No bulk download; scrape the [search](https://ttbonline.gov/colasonline/publicSearchColasBasic.do) or use a paid scraper. Flat label art, not bottle photos. COLA Cloud's [HF/Kaggle sample](https://huggingface.co/datasets/cola-cloud/whiskey-beer-and-california-wine-label-metadata) has no images. |
| [Roboflow 100 Wine Labels](https://huggingface.co/datasets/LibreYOLO/wine-labels) (HF mirror) | 4,643 | 12 label-field classes (maker name, vintage, appellation, ABV, logo...) | bbox, YOLO | CC BY 4.0 | **3** | Verified. Detects label regions, not brands. Useful to crop the text a VLM or OCR should read. |
| [BeerBB-1K](https://service.tib.eu/ldmservice/dataset/beerbb-1k) / [arXiv 2201.03791](https://arxiv.org/abs/2201.03791) | 5,207 | beer brands (count not stated in abstract) | brand label + bbox; Faster R-CNN + ResNet-18 hit 99.86% | paper CC BY 4.0, data licence unknown | **2** | TIB page sits behind an anti-bot wall; download not verified. German beers, single bottle per image. |
| [Kaggle Top Shelf](https://www.kaggle.com/datasets/jenlooper/bottles) | ~200+ (220 MB) | spirit types (brandy, amaretto, bourbon...) | folder per type | CC0 | **2** | Verified. Type-level, not brand. Minibar bottles. |
| [Kaggle AlcoVision](https://www.kaggle.com/datasets/masteralive06/alcovision) | 1.1 GB, Roboflow COCO export | unverified | bbox (COCO) | MIT | **2** | Verified URL; classes unknown without download (Kaggle auth). |
| [Kaggle DataCluster alcohol bottles](https://www.kaggle.com/datasets/dataclusterlabs/alcohol-bottle-images-glass-bottles) | sample of "3000+" | none (bottle only) | sparse | CC0 sample, full set paid | **1** | India, crowdsourced. Generic bottle detection only. |
| [Roboflow sujals/bottle](https://universe.roboflow.com/sujals/bottle-94xje) | 363 | 3 (Carlsberg, Gorkha, Somersby) | bbox | CC BY 4.0 | 2 | Unverified. |
| [Roboflow Waqas](https://universe.roboflow.com/rosd/waqas) / [Final_HR](https://universe.roboflow.com/rosd/final_hr) | unknown | 11 / 16 (coke, sprite, pepsi, redbull, fanta, 7up, beer can...) | bbox | likely CC BY 4.0 | 2 | Unverified. Soft drink bottles and cans by brand. |
| [Roboflow Wine Bottles](https://universe.roboflow.com/thaiduc-ukmdb/wine-bottles-bndbd) | 498 | unknown | bbox | unknown | 1 | Unverified. |
| [Roboflow alcohol-classification](https://universe.roboflow.com/alcohol-classification/bottle-classification-kgocu), [alcohol_bottle_detection](https://universe.roboflow.com/mymljourney/alcohol_bottle_detection), [Bundy-Jim-Chivas](https://universe.roboflow.com/test-rxqyj/bundy-jim-chivas), [package_rum](https://universe.roboflow.com/package-store/package_rum) | unknown | unknown (Bundy/Jim Beam/Chivas in the name) | bbox | CC BY 4.0 per snippets | 2 | Unverified. Worth one look in a browser for UK-relevant brands. |
| [Vinmonopolet open API](https://www.vinmonopolet.no/om-oss/presse/datadeling) | unverified | full Norwegian catalogue | product details | own ToS, free registration | 2 | Verified page; images in the API not confirmed. |
| [Systembolaget product list](https://www.kaggle.com/datasets/schmidtpalexander/worlds-largest-liquor-store-product-list) | none | all Systembolaget SKUs (Jan 2020) | name, volume, ABV, category | CC0 | 1 | Verified. Taxonomy only. Live API needs a key. |
| [Alcoholic drink units](https://www.kaggle.com/datasets/thedevastator/alcoholic-drink-units-volume-abv-and-brand-infor) | none | 800+ UK drinks | brand, style, volume, package, ABV, units | CC0 | 2 | Verified. A UK SKU taxonomy to seed the closed set. |
| [Untappd API](https://untappd.com/api/docs) | label image per beer | millions of beers | `beer_label`, brewery, style | Untappd ToS | 2 | Docs live; new-app approval status unverified. |
| [Wine label one-shot recognition, arXiv 2404.08820](https://arxiv.org/abs/2404.08820) | n/a (method) | n/a | ViT + batch-all triplet, 3D viewpoint augmentation from one label image | paper | **4** | The method that fits us: one catalogue image per SKU, enrol, nearest neighbour. No dataset released. |
| [Nyckel beer brand classifiers](https://www.nyckel.com/pretrained-classifiers/beer-brands-by-bottle/) | n/a | 10 (bottle), 29 (other model) | hosted model | commercial | 1 | Verified. Baseline only. |
| WISK bottle library ([claim](https://wisk.ai/vs/wisk-vs-bevspot)) | "200,000+ bottles" with images | proprietary | n/a | closed | 1 | Not published. Partender and BevSpot publish nothing. |
| [TheCocktailDB ingredient images](https://www.thecocktaildb.com/api.php) | one generic image per ingredient | ~500 ingredients | name | free API | 1 | Generic gin.png, not brands. |

Not found: any open Kaggle wine-label brand set, any public UK liquor board catalogue with images, any bar-inventory vendor bottle library. RP2K (500k retail shelf images, 2,000 SKUs, some drinks) has a dead project page.

## Closed-set size for a UK bar (estimate)

CGA data (via The Spirits Business, ~2018) puts the average UK pub at **36 spirit brands**. Add bottled/canned beer and cider (10-25), wine by the bottle (10-20), softs and mixers (10-20), liqueurs, vermouths and syrups (10-30). **A pub is roughly 80-150 SKUs; a cocktail bar 200-400.** The UK on-trade universe is far larger: Matthew Clark alone lists 2,000+ spirit lines. Plan for a per-venue gallery of ~150 SKUs enrolled at onboarding, not a global classifier.

## Recommendation
1. Do not train a brand classifier: enrol each venue's ~150 SKUs as reference images (Open Food Facts packshots, plus one phone shot per bottle at setup) and match crops by embedding (DINOv2/SigLIP), per arXiv 2404.08820.
2. Pull the 12 test items from Open Food Facts by barcode now; fall back to LCBO.dev or Wine Images 126K for anything missing. Barcode gives size and ABV for free.
3. Use the VLM only for the long tail (similarity below threshold), and treat Roboflow sets as unverified until someone opens them in a browser.
