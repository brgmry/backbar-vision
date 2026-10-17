# Open-set product ID: how many images per SKU (10 Oct 2026)

Research notes from the first morning, kept as written. The advice to use our own 7 photos as the gallery is what produced the leave-one-photo-out result of 97.6%, which used ground truth as an input and does not count (see the README).

## Short answer

- 1 in-domain image per SKU gets you ~85%, 5 gets ~87%, and it plateaus by 10 to 20 (~87.5%). That is NVIDIA's DINOv2-based retail recogniser on 315 classes ([NGC model card](https://catalog.ngc.nvidia.com/orgs/nvaie/models/retail_object_recognition)). Beyond ~5 references, extra images barely move accuracy. The backbone and the domain gap matter far more.
- The domain gap is the real cost. Matching real shelf crops to studio catalogue shots, off-the-shelf DINOv2 ViT-L gets 35.5% top-1 over ~2k products. DINOv3 at 384px gets 53.8%, Gemini-Embedding-2 68.7%, and fine-tuning on 196k real crops gets 80.7% ([Cat2Real](https://arxiv.org/abs/2607.09888)). Gallery images must come from our own camera, not from the web.
- Liquor is the hardest retail category. On RP2K, with ~250 training images per SKU, ResNet-34 gets 95.2% overall but only 77.0% on liquors ([RP2K](https://arxiv.org/abs/2006.12634)). Bottles are glossy, curved and near-identical across sizes and expressions.
- For our 12-SKU, visually distinct set: 1 to 3 crops per SKU taken from our own photos should clear 95% on brand. This is an inference: the published numbers come from galleries of 300 to 10k classes, and accuracy falls as the gallery grows. Size variants of the same brand (70cl vs 35cl) are the real failure mode. No embedding separates those reliably, so size has to come from the catalogue, from bottle geometry or from reading the label.

## Evidence

| Source | Task / data | Refs per SKU | Backbone | Result |
|---|---|---|---|---|
| [NVIDIA Retail Object Recognition](https://catalog.ngc.nvidia.com/orgs/nvaie/models/retail_object_recognition) | 315 retail classes, 180 test imgs/class | 1 / 2 / 5 / 10 / 20 / 100 | NV-DINOv2 + embedding head | 84.9 / 86.2 / 86.9 / 87.3 / 87.5 / 87.8% acc. Classification head: 90.1% |
| [Cat2Real](https://arxiv.org/abs/2607.09888) (2026) | 5,043 real crops vs catalogue of 2,155 products | catalogue packshots (~8/product) | DINOv2 L/14, DINOv3 L/16, Gemini-Embedding-2 | Off the shelf: 35.5 / 53.8 / 68.7% top-1 (61.4 / 78.1 / 89.3% top-5). Fine-tuned DINOv3-384: 80.7% top-1, 94.7% top-5 |
| [GroceryVision MPR study](https://arxiv.org/abs/2605.18029) (2026) | 409 SKUs, text gallery (1 description/SKU) | 1 | 190 open VLM encoders | SigLIP2 g-opt 384: R@1 77.0%, R@5 94.5%. PE-Core-L: 75.8%. CLIP ViT-L: 56.8%. Gains plateau at 384px |
| [DIHE, Tonioni & Di Stefano](https://arxiv.org/abs/1902.00760) | Grocery Products, ~8.4k products | 1 studio image | VGG-16 + GAN-trained embedding | 85.3% top-1, 94.8% top-5. Off-the-shelf VGG MAC: 79.2% |
| [OS2D](https://arxiv.org/abs/2003.06800) | GroZi-3.2k, one-shot detection on unseen classes | 1 | ResNet-101 | 90.6 mAP (detector + retrieval baseline: 86.8) |
| [Tur et al.](https://arxiv.org/abs/2409.14963) | MIMEX, 28 product categories | ~370 (kNN prototypes) | CLIP, DINOv2 | CLIP text zero-shot: 50.3%. CLIP visual kNN: 94.1%. Visual references beat text prompts by ~44 pts |
| [RP2K](https://arxiv.org/abs/2006.12634) | 2,000 SKUs, 500k shelf images | ~250 | ResNet-34 | 95.2% overall, 77.0% liquors |
| [Products-10K](https://arxiv.org/abs/2008.10545) | 10k SKUs, ~150k images | ~15 (long tail) | EfficientNet-B3 | 64.1% top-1 |
| [Retail-786k](https://arxiv.org/abs/2309.17164) | 3,298 entities, 786k leaflet images | ≥10 train | ConvNeXt (cls), ResNet-50 (retrieval) | 85.5% classification acc. Retrieval R@1: 44.9% |
| [Proxy metric learning, ICCS 2026](https://doi.org/10.1007/978-3-032-29909-3_33) | RP2K + sparse in-house set | sparse | metric learning | Proxy losses beat contrastive. Useful only if we fine-tune |
| Detection | | | | |
| [Grounding DINO](https://arxiv.org/abs/2303.05499) | COCO / ODinW zero-shot | 0 | Swin | 52.5 AP COCO, 26.1 AP ODinW mean |
| [OWLv2](https://arxiv.org/abs/2306.09683) | LVIS rare zero-shot | 0 (also supports image-conditioned 1-shot) | ViT | 44.6 AP rare |
| [YOLO-World](https://arxiv.org/abs/2401.17270) | LVIS zero-shot | 0 | YOLOv8 + CLIP text | 35.4 AP at 52 FPS (V100) |
| [RF100-VL](https://arxiv.org/abs/2505.20612) (NeurIPS 2025) | 100 out-of-distribution detection sets | 0 / 10 shots | various | Zero-shot: GDINO 15.7, OWLv2 13.6, Gemini 2.5 Pro 11.6, Qwen2.5-VL 5.6 mAP. 10-shot fine-tuned GDINO: 33.6. Full YOLOv8m: 56.4 |
| SKU-110K dense shelves | ~147 objects/image | n/a | n/a | No published zero-shot GDINO/OWLv2/YOLO-World number found. Supervised YOLO is the norm |
| Label reading | | | | |
| [Unitail](https://arxiv.org/abs/2204.00298) | 1,454 products, one-shot gallery | 1 | ABINet OCR + Hungarian text match | OCR word acc. 77.2%. Text-only matching: 47.8%. Visual-only (ResNet-101): 85.0%. Visual + text: 86.2%. Hard split: 58.4 to 60.5% |
| Synthetic data | | | | |
| [RPC checkout](https://arxiv.org/abs/1901.07249) | 200 SKUs, train on isolated product shots only | ~270 exemplars | FPN detector | Exemplars only: 0.01% cAcc. Copy-paste: 9.3%. CycleGAN-rendered: 45.6%. Both: 56.7% (mAP50 96.6%) |
| [Cut, Paste and Learn](https://arxiv.org/abs/1708.01642) | GMU Kitchen, AVD | n/a | Faster R-CNN | Real 86.3, synthetic 76.2, real+synth 88.8 mAP. AVD: 10% real 15.8, plus synth 43.2, beating 100% real (41.9) |
| [DIHE](https://arxiv.org/abs/1902.00760) | GAN augmentation of 1 studio image | 1 | VGG-16 | +3.0 pts top-1 (79.9 to 82.9) |
| NVIDIA (above) | 81k synthetic + 48k real | | NV-DINOv2 | Shipped production model trains on a 63% synthetic mix |
| VLMs | | | | |
| [Ramachandran et al.](https://arxiv.org/abs/2507.01955) (ICLR 2026) | ImageNet, COCO via prompt chaining | 0 | GPT-4o, Gemini 2.0 Flash, Claude 3.5 Sonnet | ImageNet: 77.2 / 74.8 / 62.9% vs 90.9% specialist. COCO AP: 31.9 / 19.9 / 14.8 vs 58.7 (DETR) |
| [FG-BMK](https://arxiv.org/abs/2504.14988) (ICLR 2026) | 12 fine-grained sets incl. Products-10K | 0 | GPT-4o, InternVL, Qwen | LVLMs do not beat tailored fine-grained models. No product-specific number published |
| [GPT-4o vs GPT-5 counting](https://arxiv.org/abs/2512.03233) | FSC-147, CARPK (dense) | 0 | GPT-4o, GPT-5 | FSC-147 MAE ~22. CARPK MAE 30 to 40; specialists win clearly on dense scenes |
| [VLMCountBench](https://arxiv.org/abs/2510.04401) | synthetic shapes | 0 | several VLMs | Reliable on one object type, fails on mixed types |
| [MeasureBench](https://arxiv.org/abs/2510.26865) | real gauge reading | 0 | Gemini 2.5 Pro, GPT-5 | 30% / 20% (relevant to fill level) |

Gaps: I found no peer-reviewed accuracy number for OCR on spirit labels specifically. I also found no frontier-VLM benchmark on SKU-level shelf ID with Claude 4.x or Gemini 3. The VLM numbers above are 2024/25 models on general tasks.

## What the evidence says

1. Visual references beat text by a wide margin (Tur: 94% vs 50%). A text-only VLM prompt is the weakest way to do ID. A VLM shown candidate reference crops is a different and stronger setup.
2. Use in-domain references, not catalogue shots. The catalogue-to-real gap costs 30 to 45 points off the shelf (Cat2Real). Our 7 photos are our gallery.
3. The backbone matters more than the number of shots. At 384px, DINOv3/SigLIP2 beat DINOv2 at 224px by ~18 pts. Going from 1 to 100 shots adds 3 pts (NVIDIA).
4. OCR on its own is weak (48%) but breaks ties (+1 to 3 pts), and those ties are exactly the size/expression variants that embeddings miss.
5. Synthetic data helps detectors, not 1-shot retrieval, in a 4-hour window. Naive copy-paste alone is poor (RPC 9%). It only pays off with realism (rendering/GAN) plus some real data. Skip it today.
6. Open-vocab detectors find "bottle" fine but degrade out of distribution (15.7 mAP zero-shot on RF100-VL). Ten labelled shots doubles it. We have 7 photos × 12 items = 84 boxes, which is enough to fine-tune or just to filter.

## Recommended pipeline for a 4-hour build

1. Detect (30 min). Grounding DINO or OWLv2 via HF `transformers`, prompt `"bottle . jar . carton . jug"`, with NMS. Fallback: ask Claude for boxes (weaker at localisation, COCO AP 15 to 32 for VLMs), or hand-box the 7 photos. Sort left to right.
2. Gallery (20 min). Crop each of the 12 items from 1 to 3 labelled photos. Hold out the rest for test, per CLAUDE.md.
3. Embed + kNN (40 min). DINOv2 ViT-L/14 or SigLIP2 SO400M at 384px, cosine similarity, top-3 candidates per crop. Expect brand accuracy above 95% on 12 distinct SKUs.
4. Verify / read (60 min). Send the crop plus the top-3 gallery crops and their catalogue rows to Claude and ask it to pick one or say "none" (open set). It reads the label for size/ABV, and `rapidfuzz` matches the OCR text against `inventory.csv`. This mainly covers unseen SKUs and size variants.
5. Open set. If the top-1 cosine falls below a threshold tuned on held-out crops, flag the crop as unknown and route it to the VLM without candidates.
6. Eval (30 min). Run leave-one-photo-out: gallery from k photos, test on the rest, at k = 1, 2, 3. Report ID accuracy per k against the VLM-only baseline. That gives our own shots-vs-accuracy curve.

Skip: synthetic data, fine-tuning embeddings, SKU-110K detectors. None of them pays back inside 4 hours on 12 SKUs.
