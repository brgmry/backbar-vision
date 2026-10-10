# Retail product recognition: datasets and models (10 Oct 2026)

URLs checked with curl on 10 Oct 2026. "Gated" = form, email or EULA before download. Roboflow returns 403 to curl (bot wall), loads in a browser. Drink share marked ~ is estimated from the category list, not counted. Usefulness is for this project: open-ended bottle ID (brand, product, size) from a back-bar photo, in a 5-hour hack.

## Datasets

| Dataset | URL (status) | Images | SKUs / classes | Labels | Licence | Download | Drinks / bottles | Use (1-5) |
|---|---|---|---|---|---|---|---|---|
| Open Food Facts | [data](https://world.openfoodfacts.org/data) (200), [HF](https://huggingface.co/datasets/openfoodfacts/product-database) (200), [S3](https://registry.opendata.aws/openfoodfacts-images) | millions (product photos, multi-view) | ~4M products by barcode | brand, product name, **quantity (size)**, category, OCR | images CC BY-SA, data ODbL | multi-TB bucket; filter by category via API/parquet | beverages and alcoholic-beverages categories, real packshots incl. spirits; ~10-15% beverages | **5** reference gallery with brand + size |
| WineSensed | [DTU](https://data.dtu.dk/articles/dataset/WineSensed_Learning_to_Taste_A_Multimodal_Wine_Dataset/23376560) (200 browser, 403 curl), [project](https://thoranna.github.io/learning_to_taste/) (200) | 897k label photos | 350k vintages | wine name, producer, year, region, ABV, price | CC BY-NC 4.0 | 10 x 3.5 GB + 126 MB metadata (~35 GB) | 100% wine bottles/labels (Vivino user photos) | 4 for wine only |
| RP2K | [arXiv](https://arxiv.org/abs/2006.12634), [OpenI mirror](https://openi.pcl.ac.cn/thomas-yanxin/rp2k/datasets) (200); pinlandata.com link **404** | 500k+ shelf crops | ~2,000 | SKU class, size, shape, flavour | research | ~10 GB (mirror, needs OpenI login) | Chinese retail; drinks and baijiu a large share (~30-40%) | 3 fine-tune embeddings on shelf crops |
| Products-10K | [site](https://products-10k.github.io/) (200), [Kaggle](https://www.kaggle.com/competitions/products-10k) (200) | ~150k | 10,000 (JD.com) | SKU id, group (hierarchy) | research, Kaggle rules | ~20 GB | food/drink, liquor present; ~15-20% | 3 metric-learning pretraining |
| RPC | [site](https://rpc-dataset.github.io/) (200), [Kaggle](https://www.kaggle.com/datasets/diyer22/retail-product-checkout-dataset) (200) | 83,739 (53,739 single-product + 30k checkout) | 200 SKUs, 17 meta-cats | boxes, SKU class, counts | CC BY-NC-SA 4.0 | 15 GB | drink, alcohol, milk, instant drink: ~4/17 cats (~25%) | 2 top-down checkout, not shelves |
| SKU-110K | [GitHub](https://github.com/eg4000/SKU110K_CVPR19) (200), [tar](http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz) (200, 12.2 GB) | 11,762 | 1 class ("object") | 1.73M boxes, dense shelves | research (Ultralytics yaml AGPL-3.0) | 12.2 GB | mixed shelves, drinks aisles included; no SKU labels | 3 class-agnostic "every item" detector |
| Unitail | [site](https://unitedretail.github.io/) (200) | ~11.7k det + OCR set | 1,454 products (OCR/matching) | quadrilateral boxes (1.8M), 21k text transcriptions | academic only | **Gated** (email for detection; OCR on Drive) | mixed, some drinks | 3 detect + read label text |
| Locount | [GitLab](https://isrc.iscas.ac.cn/gitlab/research/locount-dataset) (200) | 50,394 | 140 | boxes + instance counts (1.9M) | research | ~60 GB (Baidu/Drive) | Drinks is 1 of 9 supers (~10%) | 2 |
| MVTec D2S | [site](https://www.mvtec.com/company/research/datasets/mvtec-d2s) (200) | 21,000 | 60 | instance masks | CC BY-NC-SA 4.0 | **Gated** (form), ~13 GB | few bottles (~5%) | 1 |
| Retail-786k | [GitHub](https://github.com/retail786k/retail786k-dataset) (200), [Zenodo](https://zenodo.org/record/7970567) | 786,179 leaflet crops | 3,298 entities / ~18k GTINs | entity class | CC BY-NC-ND 4.0 | 22.6 GB (256px) / 65 GB (512px) | EU retailers; beer, wine, spirits common in leaflets (~15%) | 2 packshot-style, not shelf |
| Domain-adaptive synthetic shelves (SG3k, SPS8k) | [Zenodo](https://zenodo.org/records/7750242) (200) | 10k-16k per set | up to 8,112 GTINs | YOLO boxes, GTIN class | CC BY 4.0 | 8 archives, 7-12 GB each | supermarket mix; share not stated | 2 |
| Amazon Berkeley Objects | [site](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) (200) | 398k catalogue + 360 spins + 3D | 147,702 products | brand, title, dimensions, 3D | CC BY 4.0 | 581 GB total; small images 3.25 GB | home goods; drinks negligible (<1%) | 1 |
| Grozi-120 | [site](http://grozi.calit2.net/grozi.html) (200, self-signed cert) | 676 in vitro + 11k in situ frames | 120 | class per frame/box | research | ~2 GB | some cans/bottles (~15%) | 1 dated (2007) |
| Grozi-3.2k / Grocery Products, GP-180 | no live official host; unibo GP-180 page **dead**; [Roboflow copy](https://universe.roboflow.com/feruricar/grozi-3.2k-testing-h9vap) (403 curl) | 3.2k (8.3k product refs, 680 shelves) | 80 food / 180 (GP-180) | class, shelf boxes | research | ~1 GB | some drinks | 1 |
| Freiburg Groceries | [GitHub](https://github.com/PhilJd/freiburg_groceries_dataset) (200) | 4,947 | 25 | image class | research | ~0.5 GB | juice, water, soda, milk, coffee, tea (~25%) | 1 category-level only |
| Grocery Store (Klasson) | [GitHub](https://github.com/marcusklasson/GroceryStoreDataset) (200) | 5,125 | 81 | fine + coarse class, iconic image | research | ~1 GB | cartons (juice, milk) ~20% | 1 |
| Roboflow 481 spirits | [Universe](https://universe.roboflow.com/bottle-7lzmp/481) (403 curl) | 479 | 22 spirit brands | instance masks | likely CC BY 4.0 | small | 100% spirits (Jack Daniels, Jameson, JW Black...) | 4 sanity-check set |
| Roboflow Liquor-data | [Universe](https://universe.roboflow.com/elfailmortalworks/liquor-data-ucksh) (403 curl) | ~1k | spirit type x size | boxes | per-project | small | 100% spirits, size variants | 3 |
| RF100 wine-labels | [HF](https://huggingface.co/datasets/LibreYOLO/wine-labels) (200) | 1k-10k | label fields (ABV, appellation...) | boxes | CC BY 4.0 | small | 100% wine labels | 2 label-region detector |
| UMRetail (2025) | DOI 10.1109/ICVRV67992.2025.00065, no public link found | 17,697 | 3,812 | masks, boxes, vacancy, text descriptions | unknown | **not available** | shelves; share unknown | 0 |
| Retail-YU (2025) | [Mendeley](https://data.mendeley.com/datasets/mmcf24t9vv/1) (**removed by author**) | ~103k | 1,505 | boxes, shelf + web domains | n/a | n/a | n/a | 0 |

## Pretrained models

| Model | URL (status) | What it is | Licence | Use (1-5) |
|---|---|---|---|---|
| Cat2Real-DINOv3-384 | [HF](https://huggingface.co/zhanganyi88/Cat2Real-DINOv3-384) (200), [paper](https://arxiv.org/abs/2607.09888) Jul 2026 | DINOv3 ViT-L fine-tuned to match in-store crops to catalogue packshots; zero-shot to unseen SKUs | Apache-2.0 | **5** crop-to-reference retrieval |
| Marqo e-commerce embeddings B/L | [L](https://huggingface.co/Marqo/marqo-ecommerce-embeddings-L), [B](https://huggingface.co/Marqo/marqo-ecommerce-embeddings-B) (200) | SigLIP fine-tuned on millions of e-com products; image and text in one space | Apache-2.0 | **5** text query "Tanqueray 70cl" against crops |
| NVIDIA TAO retail_object_detection (binary) | [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_detection) (200) | class-agnostic "retail item" detector | NVIDIA Model EULA (gated by NGC login) | 4 one-box-per-bottle |
| NVIDIA TAO retail_object_recognition | [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/retail_object_recognition) (200) | embedding head, 2048-d, few-shot with 20-30 refs/class; 315 checkout classes | NVIDIA Model EULA (gated) | 3 |
| DINOv2 | [HF](https://huggingface.co/facebook/dinov2-base) (200) | generic self-supervised features, strong instance retrieval | Apache-2.0 | 4 |
| DINOv3 | [HF](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) (200) | stronger features | DINOv3 licence, **gated** (manual approval) | 3 (use Cat2Real instead) |
| SigLIP2 so400m | [HF](https://huggingface.co/google/siglip2-so400m-patch14-384) (200) | zero-shot image-text, reads label text well | Apache-2.0 | 4 |
| OpenCLIP ViT-H/14 LAION-2B | [HF](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) (200) | zero-shot, knows big spirit brands | MIT | 3 |
| Grounding DINO / OWLv2 | [GDINO](https://huggingface.co/IDEA-Research/grounding-dino-base), [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) (200) | open-vocab "bottle" boxes | Apache-2.0 | 4 detector stage |
| SKU-110K RetinaNet | [GitHub](https://github.com/eg4000/SKU110K_CVPR19) (Drive link in README) | dense shelf item detector | research | 2 (Keras, old) |
| beer-whisky-wine ViT | [HF](https://huggingface.co/firas-spanioli/beer-whisky-wine-detection) (200) | 3-class classifier | none stated | 1 |

## Recommendation
1. No dataset here has back-bar spirits at SKU level with size, so do not train: detect (Grounding DINO or OWLv2 "bottle") then retrieve each crop with Cat2Real-DINOv3 or Marqo e-commerce-L against a reference gallery.
2. Build the gallery from Open Food Facts (beverages + alcoholic beverages, which carry brand and quantity) plus 3-5 phone shots per bottle in `inventory.csv`; WineSensed only if wine matters.
3. Use Roboflow 481 spirits as a quick external sanity set; skip RPC, D2S, ABO, Grozi, Freiburg (wrong domain or too small), and treat Unitail/D2S/NVIDIA as gated.
