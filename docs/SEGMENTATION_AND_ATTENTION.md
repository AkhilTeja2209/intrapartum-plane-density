# U-Net segmentation and the anatomical attention check

Two additions, and they turn out to answer each other's question. The
segmentation model says *where* the anatomy is; the attention check asks
whether the classifiers were ever looking there.

The first version of this page reported one U-Net, at one seed, scored at its
own input resolution. Eleven arms later, two of its three headline numbers have
moved and one of its conclusions is withdrawn. §6 says which.

The attention check also turned out to have a second instrument. The test set
contains two acquisition campaigns three years apart, and no model transfers
between them — the best-scoring one least of all, at chance level on the older
campaign (§4.4).

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

## 2. Segmentation

`ResUNet` — a U-Net decoder on the same ImageNet ResNet-18 encoder the
classifier uses, so any difference is the decoder and the objective, not the
backbone. Cross-entropy + soft Dice over the foreground, 2,575 training masks,
selected on validation Dice.

### 2.1 The frames were cached at half the masks' resolution

Worth stating before any number, because it invalidates the obvious experiment.

The source videos are **512×512** and the shipped masks are **512×512**, but
`src/extract_frames.py` caches every frame pre-resized to a 256px short side —
the right trade for 65,531 frames feeding a classifier, and the wrong one for a
segmenter. Two consequences:

* A "384px" arm trained on the cached frames is being fed **interpolated
  pixels**. It cannot recover detail that was discarded at decode time.
* Dice computed at 256 is measured against a nearest-neighbour **downsampled**
  mask. Published Dice on this corpus is computed at native resolution, so a
  256px number is not comparable to 0.893–0.930.

`tools/extract_mask_frames.py` re-decodes just the 2,915 masked frames at
512×512 as lossless PNG — 209 MB and 41 seconds, because only those frames need
it — and `tools/unet_compare.py` re-scores every arm one way: input resized
from the native frame to whatever that arm trained at, output resampled back up
to 512, Dice against the untouched 512 mask. Every number below is that.

Re-scoring the original run this way moves it from 0.8699 to **0.8661**. The
earlier figure was optimistic by 0.004, entirely through the evaluation grid.

### 2.2 Eleven arms

| arm | input | params | val Dice | test PS | test FH | **test mean** |
|---|---|---:|---:|---:|---:|---:|
| `ens_native_sizes` | 3× ensemble | 43.0M | 0.8741 | 0.8531 | 0.9188 | **0.8859** |
| `ens_best_mixed` | 3× ensemble | 53.1M | 0.8751 | 0.8404 | 0.9167 | **0.8785** |
| `ens_seeds` | 3× ensemble | 43.0M | 0.8730 | 0.8342 | 0.9147 | **0.8745** |
| `native384_s0` | 384px native | 14.3M | 0.8686 | 0.8522 | 0.9119 | **0.8821** |
| `native256_s0` | 256px native | 14.3M | 0.8722 | 0.8331 | 0.9132 | **0.8731** |
| `base_s2` | 256px cached | 14.3M | 0.8675 | 0.8242 | 0.9160 | **0.8701** |
| `size384_s0` | 384px cached | 14.3M | 0.8710 | 0.8274 | 0.9078 | **0.8676** |
| `base_s0` | 256px cached | 14.3M | 0.8700 | 0.8352 | 0.8970 | **0.8661** |
| `native512_s0` | 512px native | 14.3M | 0.8701 | 0.8252 | 0.9053 | **0.8652** |
| `base_s1` | 256px cached | 14.3M | 0.8724 | 0.8145 | 0.9122 | **0.8633** |
| `resnet34_s0` | 256px cached | 24.4M | 0.8690 | 0.8103 | 0.9123 | **0.8613** |
| `noaug_s0` | 256px cached | 14.3M | 0.8718 | 0.8160 | 0.9050 | **0.8605** |
| `scratch_s0` | 256px cached | 14.3M | 0.8634 | 0.7999 | 0.9144 | **0.8571** |
| `ce_only_s0` | 256px cached | 14.3M | 0.8612 | 0.8109 | 0.8980 | **0.8545** |

**The bar is the seed spread: 0.8665 ± 0.0034** over `base_s0/1/2`, which
differ only in the random seed. Anything inside ±0.007 of 0.8665 is noise.

What clears it:

* **Native-resolution input, +0.0156 at 384px** — 4.6 seed-sigma, the largest
  single-model effect in the table, and the only architectural-sounding one
  that is really about the data. `native256_s0` gains +0.0066 on input source
  alone, at identical input size: re-decoding and resizing once beats resizing
  at decode time and again at load time, through a JPEG.
* **Ensembling, +0.008 to +0.019.** Averaging three seeds gives 0.8745;
  averaging the three native-resolution arms gives **0.8859**, the best result
  here and within 0.007 of the bottom of the published range.

What does not clear it, and what actively hurts:

* **More resolution is not monotonic.** 384px native beats 256px native by
  0.009, and 512px native is *worse than both* at 0.8652 — below the native
  256 arm it should dominate. With 2,575 masks, the extra capacity of a
  512-pixel field costs more than the detail buys.
* **ResNet-34 loses 0.005** against ResNet-18 on the same data. More encoder is
  the wrong direction at this sample size.
* **Augmentation is worth about 0.006** — real but small, and roughly the size
  of the seed spread.
* **Dropping the Dice term costs 0.012** and **training from scratch costs
  0.009.** Both are outside the spread, so the two least fashionable choices in
  the recipe are the two that are actually carrying it.
* **Test-time augmentation was rejected on validation** (−0.0007) and would
  have gained +0.0021 on test. One more coin-toss-sized effect; §2.3 explains
  why validation could not tell.

The symphysis scores 6–9 points below the head in every arm, as it does in
every published result: it is ~1.5% of a frame against the head's ~14%.

### 2.3 The validation split cannot pick a winner

40 masks, from 40 videos. Across the 11 single arms, validation Dice and test
Dice correlate at **Pearson r = +0.46, Spearman r = +0.27**. The arm that wins
on validation is `base_s1`; the arm that wins on test is `native384_s0`; and
**selecting on validation costs 0.0187 Dice**, which is five times the seed
spread and larger than every effect in §2.2 except native resolution itself.

This is not a detail. It means that for this corpus, at this validation size,
"best on validation" is close to a coin toss — and it is the same thinness that
makes §3's rule selection flip between seeds. Any future arm here needs a
bigger selection set before its result means anything; the cheapest fix is to
hold out training videos rather than rely on the 40 the dataset provides.

### 2.4 Mask encodings differ between splits

Worth recording because nothing documents it and it fails silently: train masks
use `{0, 7, 8}`, val and test masks use `{0, 127, 255}`. Reading a test mask
with the train encoding yields an all-zero mask — every area comes out zero and
no error is raised. `src/segmentation.py` carries both encodings and raises on
an unrecognised value, and the synthetic smoke test now writes both so CI
exercises the decoder on each.

Class identity was established by area, not assumption: across all splits the
smaller class covers ~1.5–1.8% of the frame and the larger ~13–15%, so small =
pubic symphysis, large = fetal head.

---

## 3. Can the U-Net classify? — the earlier answer was wrong

Run a segmenter over the whole test set — 8,665 frames, most of them nothing
like what it was trained on — and score each frame by how much anatomy it
found. Threshold on validation, freeze, score on the identical labels the
ResNet is scored on. Five ways of collapsing two segmented areas into one plane
score are defensible, so all five are computed in the same forward pass:

| rule | score |
|---|---|
| `min` | the smaller of the two areas — a standard plane needs *both*, so an AND |
| `geomean` | the geometric mean, a softer AND |
| `ps` / `fh` | one structure alone, to show how much is just "is there a head" |
| `sum` | the two added |

The first version of this page reported **0.607** for one model, at one seed,
with the rule fixed to `min` in advance, and concluded that the segmenter
"matches the dense frame classifier". **That conclusion does not survive eleven
arms.**

| | test macro-F1 across the 11 single arms |
|---|---|
| rule and threshold chosen on validation | **0.530 ± 0.039**, range 0.472–0.603 |
| rule fixed to `min` in advance (the original protocol) | **0.549 ± 0.099**, range 0.407–0.688 |
| for comparison: ResNet-18 `dense_all` | 0.610 ± 0.032 |
| for comparison: ResNet-18 `sparse_k1` | 0.682 ± 0.053 |

The three base seeds alone — identical data, identical configuration, different
random seed — give 0.577, 0.494 and 0.498 under validation selection, and
0.686, 0.456 and 0.479 with the rule fixed. **The original 0.607 was the top of
that distribution, not its centre.** The honest statement is that the
area-derived classifier lands around 0.53, roughly 0.08 below the dense frame
classifier and 0.15 below the best one, and that a single run of it carries
almost no information.

Two things are worth keeping from the wreckage.

**The instability is in the threshold, not the ranking.** The same models scored
threshold-free give AUPRC **0.757 ± 0.024** — a coefficient of variation of
0.031 against 0.074 for validation-selected macro-F1 and 0.181 for the fixed
rule. The segmenter's area ordering of frames is stable; converting that
ordering into a decision on a 40-video validation split is what is not. That is
the same 40 videos as §2.3, failing the same way.

Note also that choosing the rule on validation *reduces* the variance
(0.039 against 0.099) while slightly reducing the mean. Validation selection
here is bad at finding the best configuration and good at avoiding the worst —
which is the opposite of how it is usually sold.

**The best configurations do reach the frame classifier.** `native384_s0`
reaches 0.603, and the three-way native-resolution ensemble reaches **0.622**,
both inside `dense_all`'s 0.610 ± 0.032. So "a segmenter trained with no
negatives at all can classify about as well as a frame classifier trained on
53,996 labelled frames" is still defensible — for the best segmenters, as a
statement about the top of a wide distribution, which is a much weaker claim
than the one made before.

The error profile is unchanged and is the real story: `native384_s0` runs at
**recall 0.976 against specificity 0.290**. It says "standard plane" to almost
everything, which is exactly what a model that has only ever been shown
standard planes should do. It is not a plane detector; it is an anatomy
detector being asked a question it was never supervised for.


---

## 4. The attention check

`src/gradcam.py` computes the **anatomical attention ratio**: the fraction of
Grad-CAM mass falling inside the PS+FH mask, divided by what an equal-area
random region would collect. 1.0 means no anatomical preference at all.

### 4.1 Ground-truth masks only exist for positives — predicted masks fix that

Masks ship only for standard planes, so the check could originally only be run
on positives. That leaves the obvious alternative hypothesis untested: a model
might attend to anatomy when anatomy is present and to scanner furniture when
it is not, and score well either way.

`--unet` closes the gap by having the segmenter predict a mask for any frame.
Predicted masks are not ground truth, so the substitution is validated first,
on the 113–120 frames that have both:

| model | attention, ground-truth masks | attention, predicted masks | r | mean abs diff |
|---|---:|---:|---:|---:|
| ResNet-18 `dense_all` | 1.941 | 1.946 | 0.931 | 0.108 |
| ResNet-18 `sparse_k1` | 1.835 | 1.856 | 0.939 | 0.110 |
| CNN-BiLSTM, no splice | 1.306 | 1.224 | 0.907 | 0.114 |
| CNN-BiLSTM, spliced | 1.108 | 1.114 | 0.936 | 0.076 |

Per-frame correlation 0.91–0.94 and group means within 0.08. Close enough to
use on frames that have no ground truth, and the ground-truth column is the one
to quote where it exists.

### 4.2 Attention on both classes

240 frames per model, 120 of each class, predicted masks throughout. `n` is
how many produced a usable ratio; the rest are counted, not dropped (§4.3).

| model | macro-F1 | standard (n) | non-standard (n) |
|---|---:|---:|---:|
| ResNet-18 `dense_all` | 0.579 | **1.96** (120) | 1.47 (118) |
| ResNet-18 `sparse_k1` | 0.687 | **1.88** (120) | 1.51 (114) |
| CNN-BiLSTM, no splice | 0.650 | 1.20 (113) | 1.45 (89) |
| CNN-BiLSTM, spliced | **0.670** | **1.12** (120) | **1.04** (93) |

**The headline finding survives, and is now stronger.** The frame-wise models
read anatomy; the temporal models largely do not. The best-scoring temporal
model attends to the anatomy least, on *both* classes, and splicing made
attention worse rather than better — a second line of evidence, independent of
the unspliced ablation in the README's Results section, that its advantage is
not better frame understanding.

What is new is the per-class split. The frame models drop from ~1.9 on standard
planes to ~1.5 on non-standard ones, and stay clearly above chance on both.
That is the expected shape: on a non-standard frame there is less anatomy to
find and the predicted mask is less trustworthy, but the evidence the model
uses still concentrates where the anatomy is. The spliced CNN-BiLSTM sits at
1.12 and 1.04 — indistinguishable from no anatomical preference on either class.

**The segmenter drew anatomy on every single frame.** Across all 240 frames of
all four runs, zero had an empty predicted mask — including all 120
non-standard ones. That is §3's over-calling, measured directly rather than
inferred from a confusion matrix.

### 4.3 The frames with no evidence at all

A Grad-CAM map is ReLU'd, so a model with no positive evidence for "standard
plane" anywhere in a frame produces an all-zero map and no ratio. How often
that happens is itself a result:

| model | empty maps, standard | empty maps, non-standard |
|---|---:|---:|
| ResNet-18 `dense_all` | 0 / 120 | 2 / 120 |
| ResNet-18 `sparse_k1` | 0 / 120 | 6 / 120 |
| CNN-BiLSTM, no splice | 7 / 120 | **31 / 120** |
| CNN-BiLSTM, spliced | 0 / 120 | **27 / 120** |

The temporal models have no positive evidence anywhere on about a quarter of
non-standard frames; the frame models almost never do. That is not a defect —
having no evidence for "standard plane" on a non-standard frame is correct
behaviour — but it does mean the non-standard means above are computed over the
frames where the model *did* find something, i.e. with the most confidently
negative frames removed.

That selection effect is the right way to read the one number in §4.2 that
otherwise looks like a story: the unspliced CNN-BiLSTM's non-standard mean of
1.45 sits above its standard mean of 1.20, but its median runs the other way
(1.17 against 1.29) and it is computed over just 89 of 120 frames. It is a long
right tail on a thinned sample, not a finding.

### 4.4 The same question asked with a different instrument: it does not transfer

Grad-CAM asks whether the model looks at the anatomy. The other way to ask is
whether it survives a change of acquisition conditions. Earlier versions of
this page promised a leave-one-centre-out split for that and said it "still has
not been run", which implied pending work.

**It cannot be run.** IUGC2024 ships no centre or hospital column — not in
`index.csv`, not in the dataset's own `*_info.csv` files, not in its README —
and all 774 videos are re-encoded to an identical 512×512, 24 fps, single-codec
container, so no device signature survives in the metadata either. The entire
training set is one campaign, so there is nothing to hold out.

What the release does carry is acquisition date, in the filenames, and it
separates the official splits completely:

| split | n | filename convention | dates |
|---|---:|---|---|
| train | 434 | `20190726T095643_0` | Jul 2019 – Jan 2020 |
| test | 93 | `202012132137152159400I1` | Dec 2020 – Jul 2021 |
| test | 207 | `240422145736` | Apr 2024 – Jul 2024 |
| val | 40 | `test_10_80` (renamed) | unknown |

Two things follow. Every number in this project is already measured under a
1.5-to-4.5-year acquisition shift, because the project uses these splits
unchanged. And the test set holds two campaigns three years apart, so they can
be scored separately — which asks what the per-centre split was meant to ask,
from predictions that already exist. `tools/per_campaign.py` does this.

Read AUROC, not macro-F1: the campaigns have different class balance
(prevalence 0.68 against 0.52), so neither macro-F1 at one frozen threshold nor
AUPRC — whose baseline *is* the prevalence — can be read across them, while
AUROC is prevalence-independent.

Intervals resample **examinations, not clips.** Several clips can come from one
examination, sharing a patient, a session, an operator and a machine. The 2021
campaign states this in its filenames (a 20-digit examination stamp then
`I1`..`I6`), so its 93 clips are 33 examinations; 2024's clips are a median of
1.3 minutes apart and group into 84 sessions at a 10-minute rule. Resampling
clips treats correlated units as independent and reports intervals too narrow.

| arm | overall F1 | AUROC 2021 (33 exams) | AUROC 2024 (84 sessions) | gap [95% CI] |
|---|---:|---:|---:|---:|
| ResNet-18 `sparse_k1` (shipped) | 0.687 | 0.699 [0.615–0.779] | 0.861 [0.835–0.886] | **+0.162** [+0.079, +0.252] |
| ResNet-18 `dense_all` | 0.579 | 0.630 [0.550–0.708] | 0.813 [0.781–0.843] | **+0.183** [+0.098, +0.267] |
| CNN-BiLSTM, spliced (best overall) | 0.670 | **0.579 [0.495–0.667]** | 0.844 [0.797–0.885] | **+0.266** [+0.177, +0.357] |
| CNN-BiLSTM, no splice | 0.650 | 0.592 [0.502–0.673] | 0.787 [0.747–0.829] | **+0.195** [+0.103, +0.294] |

**Performance does not transfer across acquisition campaigns.** All four gaps
exclude zero, and this is not a thresholding artifact: tuning the threshold per
campaign — on test, as a diagnostic only — still leaves 2021 at 0.57–0.65
macro-F1 against 0.72–0.80 on 2024. Ranking degrades, not just calibration.

**It holds for every arm in the project, without exception.** Scored across all
25 arms that have saved test predictions — every sampling condition, seed and
architecture:

| | 2021 | 2024 |
|---|---:|---:|
| mean AUROC | 0.632 | 0.810 |
| range | 0.521 – 0.770 | 0.678 – 0.865 |

The gap is positive in **25 of 25** arms (mean +0.179, median +0.184, range
+0.047 to +0.299), and 9 of the 25 fall below 0.60 AUROC on the 2021 campaign.
The best 2021 result anywhere in the project is 0.770. A unanimous 25-of-25
sign is not a property of any one model; it is a property of the shift.

**And it is not simple temporal drift.** Training is 2019, so the 2021 campaign
is nearer to it in time than 2024 is — yet 2021 is the one that fails, by
0.18 AUROC on average. Whatever separates these campaigns, it is not elapsed
time since training, which is what makes "something about the 2021 acquisition
setup" the more plausible reading.

**The best-scoring arm is the worst offender.** The spliced CNN-BiLSTM leads
overall at 0.670 and collapses hardest: AUROC 0.576 on the 2021 campaign, with
an interval of [0.496, 0.658] that **includes 0.5**. On those 93 videos it is
statistically indistinguishable from a coin flip.

That converges with §4.2 from the opposite direction. The spliced model showed
the least anatomical attention (1.12/1.04, no measurable preference) *and* now
shows the worst cross-campaign transfer. Two independent instruments, one
conclusion: its advantage is not better frame understanding. This is the third
line of evidence, after the unspliced ablation in the README's Results section
and the attention gap.

Why the headline numbers looked healthy: 2024 is 207 of the 300 test videos, so
they are dominated by the campaign every model handles well. The 2021 third is
0.16–0.27 AUROC harder for every arm.

Calibration transfers even worse than ranking. On 2021 the macro-F1-optimal
threshold sits at 0.87–0.99 against the 0.16 frozen on validation, and at the
frozen threshold recall runs 0.92–0.99 at ~0.70 precision — the models call
very nearly every 2021 frame a standard plane.

### 4.5 What this still does not show

On negatives, the "anatomy" being measured against is a prediction from a
segmenter that has never seen a negative and over-calls them (§3). So the
non-standard column answers "does the classifier look where the segmenter
thinks anatomy is", which is weaker than what the standard column answers. The
r ≈ 0.93 validation covers positives only, because that is where ground truth
exists.

The per-campaign result establishes that performance does not transfer; it does
not isolate the scanner as the cause. The campaigns are three years apart and
may differ in patient population, operator and protocol as well as equipment,
and a per-centre split that would separate those is not available at all. What
raises the shortcut reading above speculation is the correlation with §4.2:
the arm with the least anatomical grounding degrades the most. That is
consistent with a shortcut, not proof of one.

The 40 validation videos are also renamed, so their provenance is unknown,
which compounds the thin-validation problem in §2.3: the threshold and the
arm are both selected on 40 videos that cannot be placed in either campaign.

Finally, the 2024 session grouping is a 10-minute-gap heuristic, not a stated
fact like the 2021 examination prefix. If 2024's true sessions are coarser, its
intervals are still slightly narrow — but its interval is the tight one in
every row above, so the gaps would widen away from zero, not toward it.


---

## 5. Both models in the browser

The demo now runs the cascade: the classifier scores the frame and shows its
Grad-CAM, then the U-Net outlines both structures and gives its own
area-derived opinion, which the page labels as such and prints the
recall/specificity of next to it. When the two stages disagree the page says
so, because that disagreement is the cascade's error budget made visible.

The segmenter is a second 57 MB download, so it is fetched only on request.

Two things came out of making it work that are worth recording.

### 5.1 int8 is 4x smaller, free, and unusable

Dynamic int8 quantisation takes the graph from 57.3 MB to 14.4 MB at no
measurable cost — mean foreground Dice 0.8740 against 0.8741 on the
verification frames. It then fails to load:

```
Can't create a session. ERROR_CODE: 9, ERROR_MESSAGE: Could not find an
implementation for ConvInteger(10) node with name '/stem/stem.0/Conv_quant'
```

onnxruntime-web's WASM backend has no `ConvInteger` kernel, and onnxruntime's
`quantize_dynamic` only emits the QOperator form, so there is no QDQ variant to
fall back on. The fp32 graph ships. `--quantise` is kept in the exporter
because the measurement is worth having and the small graph is usable outside
a browser.

### 5.2 Pillow does not resample in floating point

The page reimplements Pillow's bilinear geometry so the browser sees what
torchvision saw. That was already true of the *geometry*; it was not true of
the *arithmetic*. Pillow quantises each normalised weight to a 22-bit
fixed-point integer, accumulates in integers with a half-LSB offset, and shifts
back:

```
k[i] = round(w[i] * (1 << 22))
acc  = (1 << 21) + sum(pixel[i] * k[i])
out  = clip8(acc >> 22)
```

Rounding a float sum instead is not equivalent. On the 256 -> 384 upscale the
segmenter needs, a third of output pixels land on an exact half-way blend of
two neighbours, and the two roundings disagree on **14.5% of pixels** by up to
two 8-bit levels. End to end that moved the segmenter's area score **2.6e-3**
away from PyTorch.

This was invisible until now because the classifier *downscales*, where ties
are rare — the float version matched Pillow on all but ~0.15% of pixels there.
The segmenter at 256px hid it too, because that resize was a no-op. Only a
genuine upscale exposed it.

Porting the fixed-point path reproduces Pillow on **all 147,456 pixels** and
takes browser-vs-PyTorch agreement from 2.6e-3 to **3.6e-7**. Both pipelines
now self-check against recorded PyTorch references on every bundled example,
and print the comparison to the console.

---

## 6. What changed from the first version of this page

| Claim | Then | Now |
|---|---|---|
| Test Dice | 0.8699 | 0.8661 for that checkpoint at native resolution; **0.8859** for the best configuration found |
| Best setup | the only setup tried | native-resolution input at 384px, ensembled over input sizes |
| U-Net as a classifier | "matches the dense frame classifier (0.607 vs 0.610)" | **withdrawn** — see §3 |
| Attention measured on | standard planes only | both classes, via predicted masks |
| Leave-one-centre-out split | "still has not been run" | **not runnable** — no centre label exists and the videos are uniformly re-encoded; replaced by a per-campaign split (§4.4) |
| Cross-campaign transfer | not measured | **no model transfers**; the best-scoring arm is at chance on the 2021 campaign (§4.4) |

---

## Reproducing

```bash
python -m src.segmentation --dataset-root data/DatasetV3 --out data/masks.csv
python tools/extract_mask_frames.py          # masked frames at native 512px
python tools/run_unet_sweep.py               # all eleven arms, ~4 h on one GPU
python tools/unet_compare.py --classify all  # one scoring protocol for all of them

python -m src.gradcam --checkpoint results/sparse_k1__frame__seed0/best.pt \
    --unet results/unet/native384_s0/best.pt --on all --n 240
```
