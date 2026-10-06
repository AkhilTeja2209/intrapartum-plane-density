# U-Net segmentation and the anatomical attention check

Two additions, and they turn out to answer each other's question. The
segmentation model says *where* the anatomy is; the attention check asks
whether the classifiers were ever looking there.

---

## 1. Literature: where a U-Net fits

The IUGC 2024 challenge this corpus comes from is explicitly **multi-task** —
classify standard planes, segment the pubic symphysis (PS) and fetal head (FH),
then measure the angle of progression (AoP) and head-symphysis distance.

| What the field does | Source |
|---|---|
| Classic 2D U-Net was the organisers' baseline (team T0) and the backbone for teams T3, T4, T6 | [Beyond Benchmarks of IUGC](https://arxiv.org/html/2602.12922) |
| U-Net++, DeepLabV3+, LinkNet-MobileNetV2, MA-Net ensembles | same |
| DSSAU-Net (dual sparse selection attention), MFA-UNeXt (DCT decomposition) | [DSSAU-Net](https://arxiv.org/pdf/2506.03684) |
| BRAU-Net — U-shaped pure transformer with bi-level routing attention | [BRAU-Net](https://arxiv.org/pdf/2310.00289) |
| Reported Dice across prior work: **0.893–0.930** | [Beyond Benchmarks of IUGC](https://arxiv.org/html/2602.12922) |
| **Six of eight teams used a two-stage cascade**: classify first, segment only the positive frames | same |
| Dedicated segmentation corpus: PSFHS, 5,101 images from 1,175 women | [PSFHS, Sci Data](https://www.nature.com/articles/s41597-024-03266-4) |
| Video-based AoP measurement, temporal correlation between frames | [Ultrasound Video Segmentation for AoP](https://dl.acm.org/doi/10.1145/3696409.3700214) |

So the classifier already in this repo is **stage one of the standard
architecture**, and a U-Net is the natural stage two. Error accumulation across
the cascade is the limitation the challenge review names.

### What a U-Net cannot be here

Masks exist for **2,915 frames and every one is a standard plane**. There is no
"anatomy absent" supervision anywhere in the corpus. A segmenter trained on it
has never seen a negative, so whether it can classify is an empirical question,
not an architectural one. §3 measures it rather than assuming it.

---

## 2. Segmentation result

`ResUNet` — U-Net decoder on the same ImageNet ResNet-18 encoder the classifier
uses, so any difference is the decoder and the objective, not the backbone.
Cross-entropy + soft Dice over foreground only, 2,575 training masks, selected
on validation Dice.

| Class | Test Dice |
|---|---:|
| Pubic symphysis | 0.8292 |
| Fetal head | 0.9106 |
| **Mean foreground** | **0.8699** |

Against the 0.893–0.930 published range this is a little low, and the reasons
are mundane rather than interesting: 2,575 training masks against PSFHS's
5,101, a plain ResNet-18 U-Net against ensembles and transformers, no
test-time augmentation, 16 epochs. It is a working stage-two segmenter, not a
competitive challenge entry.

The symphysis scores ~8 points below the head, as it does in every published
result — it is ~1.5% of a frame against the head's ~14%.

### Mask encodings differ between splits

Worth recording because nothing documents it and it fails silently: train masks
use `{0, 7, 8}`, val and test masks use `{0, 127, 255}`. Reading a test mask
with the train encoding yields an all-zero mask — every area comes out zero and
no error is raised. `src/segmentation.py` carries both encodings and raises on
an unrecognised value.

Class identity was established by area, not assumption: across all splits the
smaller class covers ~1.5–1.8% of the frame and the larger ~13–15%, so small =
pubic symphysis, large = fetal head.

---

## 3. Can the U-Net classify?

Run the segmenter over the whole test set — 8,665 frames, most of them nothing
like what it was trained on — and score each frame by the **smaller of the two
foreground areas**. A frame showing a head and no symphysis is not a standard
plane, and taking the minimum says so. Threshold picked on validation, frozen
before test, scored on the identical labels the ResNet is scored on.

| Model | Training signal | Test macro-F1 |
|---|---|---:|
| ResNet-18, `sparse_k1` | 434 labelled frames | **0.682 ± 0.053** |
| ResNet-18, `dense_all` | 53,996 labelled frames | 0.610 ± 0.032 |
| **U-Net, area-derived** | **2,575 masks, zero negatives** | **0.607** |

The segmentation-derived classifier **matches the dense frame classifier**
(0.607 against 0.610 ± 0.032) while never having been shown a single negative
example. It is also the only one of the three whose decision is interpretable by
construction: the evidence for it is the mask it drew.

It is not better than the best classifier, and it is one seed, so treat the
ranking as provisional in exactly the way §4 of `RESULTS.md` describes. Its
error profile differs sharply though — recall 0.953 against specificity 0.310,
so it over-calls standard planes, which is what you would expect from a model
that has only ever been shown them.

---

## 4. The attention check, and what it says about the temporal arm

`src/gradcam.py` computes the **anatomical attention ratio**: the fraction of
Grad-CAM mass falling inside the PS+FH mask, divided by what an equal-area
random region would collect. 1.0 means no anatomical preference at all.

120 masked test frames per model:

| Model | Test macro-F1 | Attention ratio | Frames above 1.5 |
|---|---:|---:|---:|
| ResNet-18 `dense_all` (frame) | 0.579 | **1.94** | 90% |
| ResNet-18 `sparse_k1` (frame) | 0.687 | **1.84** | 75% |
| CNN-BiLSTM, splicing off | 0.650 | 1.31 | 29% |
| CNN-BiLSTM, splicing on | **0.670** | **1.11** | **8%** |

**The frame-wise models look at the anatomy. The temporal models largely do
not.** At 1.11 the spliced CNN-BiLSTM is within noise of chance — its evidence
is distributed almost as if the masks were irrelevant. The overlays agree with
the number: frame-wise heat sits on the symphysis and the symphysis-head
interface, while the temporal model's drifts to image corners outside both
structures.

This sharpens the Arm 2 finding rather than overturning it. `RESULTS.md` §3
already showed the temporal gain was not transition modelling, because the
unspliced ablation nearly matched the spliced model. Now there is a second,
independent line of evidence: **the temporal arm scores highest while attending
to the anatomy least**, and splicing made the attention *worse* (1.11 against
1.31). Whatever the BiLSTM is exploiting — most plausibly sliding-window logit
averaging and video-level context — it is not better frame understanding.

A model that scores well without looking at the anatomy is the failure mode
this check exists to catch. It should be reported next to the macro-F1, not
instead of it.

### Scope

120 frames per model, one seed each, and only on frames that have masks — which
are all standard planes, so this measures attention on positives only. The
leave-one-centre-out split that would separate "attends to anatomy" from
"attends to this hospital's scanner" has still not been run.

---

## Reproducing

```bash
python -m src.segmentation --dataset-root data/DatasetV3 --out data/masks.csv
python -m src.train_unet --epochs 40
python -m src.gradcam --checkpoint results/dense_all__temporal__seed0/best.pt
```
