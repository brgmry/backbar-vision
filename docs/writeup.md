# We tried to beat Claude at reading a bar. Here is why we couldn't.

Bergen and Allen, October 2026

## The bet

On a Saturday in October the two of us started on stock tracking for bars: one photo of a back bar in, and for each bottle its name and how much is left. Claude Opus 5.5 can already name many bottles if you ask it, so we set one target. Beat it by 10 points of item recall with something that runs locally, at zero API cost per photo.

We did not. This is how we found out, with the numbers, and the two things that did work.

## The number that lied

Our first local pipeline found bottles with OWLv2 and matched each crop to labelled reference crops with DINOv2. On our kitchen set (7 photos of the same 12 products, from spirits to milk and ketchup) it scored 97.6% item recall. Claude scored 59.5%.

The reference crops came from the other six photos: same bottles, same shelf, same light, labelled by us. The pipeline was looking up our answers, not recognising products. That evening we wrote a rule into the repo: ground truth is the answer, never an input. Built honestly, against an external catalogue of 30,000 products, the same idea scored 7% to 30%. The 97.6% row is still in `data/results.csv`, marked invalid.

## The honest loss

| System | Item recall | Brand recall | Runs locally |
|---|---|---|---|
| Claude Opus 5.5, open-ended | 59.5% | 38.7% | no |
| Qwen3-VL-32B, 4-bit | 38.1% | 15.6% | yes, 242 s per photo |
| Qwen3.5-9B, 4-bit | 37.7% | 20.3% | yes, 59 s per photo |
| SigLIP2 retrieval over the catalogue, plus OCR | 29.8% | 23.4% | yes |
| Gemma 4 12B, 4-bit | 6.3% | 6.9% | yes, 119 s per photo |

Kitchen set, photo as the only input. Claude is the mean of 5 runs (±1.5 points); local times are on an M5 Pro laptop.

The kitchen set is not a bar, so we also checked five real bar photos. They have no labels yet, so instead of scoring against a list we judged answers. Each system's name for a bottle was shown with the bottle's crop, without saying which system gave it, and marked right, wrong or can't tell.

| Judge | Judged | Claude right | Ours right |
|---|---|---|---|
| Allen, blind A/B, 10 Oct | 43 bottles | 18 of 23 (78%) | 6 of 24 (25%) |
| Claude Opus 5.5, with web images | 365 answers | 120 of 186 (65%) | 45 of 134 (34%) |

Shares leave out "can't tell". The second row was judged by a model, which our own rules forbid for labels (see how we measured).

The gap is wider than the table shows. On 42 bottles Claude gave only a category, "Gin" or "Red Wine", and those count as wrong. When it did name a brand it was right 120 times in 144 (83%). Our pipeline always names a brand, and was right about one time in three. On the 98 bottles both systems named and the judge could see, Claude alone was right on 36, ours alone on 4.

## Why Claude wins

A back bar is hundreds of products, many of them regional, most labels half hidden. Claude reads the visible letters, the cap, the shape and the colour together and maps them to a product it has seen described many times. A retrieval system only knows what is in its catalogue, and only as the catalogue photographed it: packshots on white, not a bottle two metres away behind a beer tap. On the kitchen set it could not even try for the milk, sauces and syrups: the catalogue had none.

To win locally you need a catalogue of every product a bar stocks, photographed the way a bar looks. Building that catalogue is the real product, and a weekend does not build it.

## Where Claude still fails

Claude is the best we found, not a solved problem:

- It invents brands it cannot read. It called a bottle of Thai Dragon "Flying Goose".
- On the kitchen set it guessed 144 of its 414 answers from the bottle's shape, and got the size right 36% of the time.
- On the 82 bar photos it listed 2,180 bottles, guessed 866 of them (40%) from shape and gave no brand for 523.
- When it names a brand, about one answer in six is wrong. It read a bottle with DEAD MAN'S FINGERS in large letters as Kraken Black Spiced Rum, another spiced rum in a dark bottle.
- Asked for millilitres left, it was off by 19 points of bottle size on average (next section).

## Fill level: Claude as a pointer, not an estimator

Fill is volume, not height, and bottles are not cylinders. So we split the job. SAM outlines the bottle. The neck counts as full and the base as empty, and the outline's width on each row gives a volume of revolution. Then we draw numbered ticks up the side of the crop and ask Claude a single question: which tick is at the liquid line? Geometry turns that tick into millilitres. If the colour barely changes across the line, we return "unsure" instead of a number.

| Method | Mean error (points of bottle size) | Answered |
|---|---|---|
| Outline + Claude names the tick | 2.1 | 2 of 4 |
| Claude estimates the ml directly | 19.0 | 4 of 4 |
| Outline + colour edge | 38.6 | 4 of 4 |
| Outline + LabPics liquid segmentation | 73.0 | 4 of 4 |
| Outline + SAM 3 "liquid" | 76.8 | 4 of 4 |

Every local liquid finder read clear glass as full. The catch: these are four photos of one 50 ml miniature holding 7.5 ml, then 15 ml, of water, and the best method answered two of them. It is a direction, not a result. A measured pour series is the next thing we would run.

## The win we did not plan

SAM 3 with the text prompt "bottle" outlined 113 bottles on one bar photo (`bar_03`) in a single call, and replaced the tiled OWLv2 detector we had been tuning. Across the 82 bar photos it drew 4,835 outlines, a median of 56 per photo. We have not counted the bottles by hand, so we can't yet say how many it missed or how many outlines are not bottles.

## What we would tell someone starting this

- For long-tail recognition the frontier model is the bar to beat, and a weekend will not beat it.
- A local system can win only with a catalogue of the actual stock and controlled capture: a fixed camera, backlight, frames over time.
- Use the big model for the narrow judgement it is good at, such as pointing at a liquid line, and do the arithmetic in code.
- Write the leakage rule down before the first result, not after.

## How we measured

The bar photos have no labels yet. Allen judged 43 bottles blind on 10 October, comparing the two systems' answers side by side. We ran out of time to judge the rest by hand, so Claude Opus 5.5 judged all 365 answers on the same five photos. It saw the bottle's crop and the top three Bing image results for the name, got the instructions the human judge had, and was never told which system gave the name.

A model grading the model we set out to beat is the bias our rule against model-made labels exists to prevent, so read that row as a cross-check, not a result. On 14 cards that Allen also judged, it agreed with him on 13, and it points the same way as the human A/B. The verdicts are in `data/test_bars/card_verdicts.csv`, and `src/score_judged.py` prints every number above from them.

Every number here comes from a committed file and a script that runs no model: `src/score.py` for predictions, `src/score_judged.py` for the judged answers and `stats()` in `src/judge_routes.py` for the A/B in `data/test_bars/judgements.csv`. Thresholds and prompts were tuned on our kitchen and pour photos, never on the bar test set, and no training image is within perceptual-hash distance 6 of a bar test photo. To rerun:

```
.venv/bin/python src/score.py data/predictions.csv
.venv/bin/python src/score_judged.py
.venv/bin/python src/score.py data/predictions_fill_pour_oracle_geometry_opus-grid.csv --alcohol
```
