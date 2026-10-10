# Prior art and datasets (10 Oct 2026)

No open dataset or repo does liquor bottle ID plus fill level. Nobody sells a fixed back-bar camera that reads fill continuously.

## Datasets
| Dataset | Size | Licence | Use |
|---|---|---|---|
| [Vector-LabPics v2](https://zenodo.org/record/4736111) | 7.9k real images | MIT | vessel + liquid masks, pretrain liquid segmentation |
| [Roboflow Bottle fill level](https://universe.roboflow.com/nourhen-chaaoui/bottle-fill-level-ethhf) | 290 | CC BY 4.0 | coarse fill classes, YOLO-ready |
| [Roboflow 481 spirits](https://universe.roboflow.com/bottle-7lzmp/481) | 479 | likely CC BY 4.0 | 22 spirit brands, instance seg |
| COCO / Open Images bottle | large | CC BY 4.0 | generic detector (stock YOLO covers it) |
| [LCDTC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10422399/) | 5.9k | on request | five fill classes, too slow to get |

Also: [Trans10K](https://github.com/Counter-F/Segment_Transparent_Objects), [TransProteus](https://github.com/sagieppel/TransProteus_Paper_Code) (synthetic), [Roboflow Liquor-data](https://universe.roboflow.com/elfailmortalworks/liquor-data-ucksh).

## Reusable code
- [sagieppel SAM2 fill-level weights](https://github.com/sagieppel/Segment_Anything2_Trained_for_Semantic_Seg_Vessell_Fill_level) (Apache-2.0): segments vessel and filled region.
- [Graduated flask YOLOv8-seg](https://github.com/syedjameel/Graduated-Flask-Liquid-Level-Detection) (MIT): fill line to mL.

## Approach
1. Segment bottle and liquid, integrate the bottle silhouette as a solid of revolution up to the liquid line, multiply by catalogue size.
2. VLM for brand/type/size only. VLMs are weak at reading measurements: [MeasureBench](https://arxiv.org/abs/2510.26865) puts Gemini 2.5 Pro at 30%, GPT-5 at 20% on real gauge images.

## Market
Partender (manual tap, ~$199-299/mo), WISK (Bluetooth scale), BevSpot (manual), Lixor.ai (voice, claims photo fill, $59-219/mo), Nectar (smart pourers). Scale weighing is the accuracy bar.

## Limits
Opaque and amber bottles hide the line. Pourers hide the neck.
