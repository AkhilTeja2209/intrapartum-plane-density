# Experimental protocol

## The question, restated precisely

> Holding the backbone architecture fixed, does the **density with which
> frames are drawn from ultrasound video** change standard-plane
> classification performance — and if so, is the gain attributable to sample
> count, or to density itself?
>
> Separately: once density is accounted for, does **explicit temporal
> modelling** add anything beyond post-hoc smoothing of frame-wise
> predictions?

## A design problem you should fix before you run anything

Your abstract proposes comparing ResNet-18 trained on **FETAL_PLANES_DB
(~12,400 images)** against ResNet-18 trained on **IUGC video frames**, holding
architecture constant so that "the dataset is the only variable."

The architecture is constant. The dataset is not the *only* other variable —
it is at least five variables at once:

| | FETAL_PLANES_DB (Burgos-Artizzu 2020) | IUGC (Zenodo 17655183) |
|---|---|---|
| Probe / approach | transabdominal | transperineal |
| Gestational stage | mid-pregnancy anatomy scan | third trimester, during labour |
| Anatomy imaged | fetal brain, abdomen, femur, thorax | pubic symphysis + fetal head |
| Label space | 6 anatomical plane classes | binary standard / non-standard |
| Clinical question | is this the right anatomical view | is this frame measurable for AoP/HSD |

Two models trained on those are not two conditions of one experiment. They are
two unrelated experiments, and any difference between them is uninterpretable:
you cannot tell whether it came from frame density, from binary-vs-6-class
difficulty, or from the fact that pubic symphysis is an easier target than
fetal femur. A reviewer will raise this immediately, and they will be right.

**The fix keeps your research gap fully intact.** Run the sparse-versus-dense
comparison *inside* IUGC. Sampling k frames per video produces a genuine
synthetic image dataset — same probe, same anatomy, same scanners, same
annotators, same label definition — and k becomes a clean, continuous dial from
"image dataset" (k=1) to "video dataset" (k=all). Every confound in the table
above is held constant by construction, which is strictly *more* control than
your original design, not less.

FETAL_PLANES_DB still earns a place in the paper, in two roles that don't
require the two datasets to be commensurable:

1. **Transfer-learning source.** Pretrain on FETAL_PLANES_DB's 6-class task,
   fine-tune on IUGC. This tests whether generic fetal-ultrasound features
   transfer to intrapartum imaging — a real question, and a positive result is
   useful to anyone with a small intrapartum dataset.
2. **External sanity check.** Report the frame-wise pipeline's numbers on
   FETAL_PLANES_DB to show it reproduces published results, establishing that
   your implementation is sound before you draw conclusions from it.

Frame that in the paper as: *prior work varies the dataset and the task
together; we construct the sparse and dense conditions from a single corpus so
that sampling density is genuinely the only thing that changes.* That is a
sharper version of the gap you already identified in your review slides.

## Design

**Fixed across every condition and both arms:** ResNet-18 (ImageNet init),
AdamW, lr 1e-4, weight decay 1e-4, cosine schedule with 1 warmup epoch, 25
epochs max, early stopping on validation macro-F1 with patience 6, batch 64
frames, identical augmentation, inverse-frequency class weighting, 224px.

**Varied:** training-frame sampling only (Arm 1) and the presence of a
temporal head (Arm 2).

**Never varied:** the validation and test sets. Sampling is applied to the
training split alone. Every condition is scored on the identical, complete,
dense test frame set — otherwise the numbers are not on the same scale.

### Arm 1 — Protocol A: the density curve

| Condition | Training frames | Meaning |
|---|---|---|
| `sparse_k1` | 1/video | synthetic image dataset, label-blind selection |
| `sparse_k1_curated` | 1/video | selection mimicking how real image datasets are built |
| `sparse_k2`, `k5`, `k10`, `k20` | k/video | intermediate density |
| `dense_stride8/4/2` | every 8th/4th/2nd | intermediate, temporally contiguous |
| `dense_all` | all | the video dataset |

Two selection strategies matter and they differ in an important way.
`uniform` picks evenly spaced frames and is label-blind, so it preserves the
natural class prior (~17% standard). `curated` picks the temporal centre of
each standard-plane run, which is what a sonographer actually does when
freezing a frame — and it produces a *positive-enriched* set (in simulation,
~76% standard at k=1). That enrichment is realistic; it is also a confound.
Inverse-frequency class weighting is applied identically everywhere to absorb
it, and `uniform` is the primary sparse condition with `curated` reported as
the realism check.

### Arm 1 — Protocol B: matched budget (this is the contribution)

`dense_matched_k5` and `dense_matched_k20` cap the dense arm to *exactly* the
frame count of the corresponding sparse arm, dropping whole videos so that
within-video density stays intact. In a synthetic run this gave 2,305 frames
from 461 videos (sparse) versus 2,289 frames from 24 videos (dense).

Reading the 2×2:

| Protocol A | Protocol B | Conclusion |
|---|---|---|
| dense wins | dense wins | density carries information beyond sample count |
| dense wins | dense loses | the gain was sample count; "video data is better" is an artefact of size |
| dense wins | tie | frames are interchangeable regardless of provenance |

**Prior work reports Protocol A and interprets it as Protocol B.** That
sentence is your paper's contribution, and Protocol B is the experiment that
earns it. Note that the dense-matched arm sees only ~24 distinct patients —
so if it loses, the honest reading is "patient diversity beats frame count,"
which is itself a clean, quotable finding.

### Arm 2 — temporal modelling

ResNet-18 encoder + BiLSTM(256), per-frame output over 16-frame clips
(~0.5 s of real time), 50% overlap in training, sliding-window logit averaging
at inference so the output is exactly one probability per test frame — the
same shape the frame-wise arm produces.

**The baseline is frame-wise ResNet-18 + post-hoc temporal smoothing, not raw
frame-wise ResNet-18.** Standard planes occur in contiguous runs, so most
exploitable temporal structure is plain label autocorrelation, and a smoother
captures it with zero parameters and zero training. In simulation with a
realistically noisy frame classifier, tuned smoothing lifted macro-F1 from
**0.834 → 0.965**. If you benchmark an LSTM against the unsmoothed 0.834 you
will "discover" a 13-point temporal gain that is entirely a post-processing
artefact — and that is exactly the error the literature you are critiquing
makes. `src/smoothing.py` gives the baseline the same tuning budget as the
LSTM (moving average, median filter, and a two-state Viterbi decode, each with
its threshold swept on validation).

Ablations: GRU vs LSTM; bidirectional vs causal (only the causal model is
deployable during live scanning, and the gap between them is a finding worth
reporting); ImageNet init vs warm-start from the trained frame-wise encoder.

## Evaluation

Splitting is by **video**, via `StratifiedGroupKFold` stratified on each
video's standard-plane fraction, with explicit assertions that no video
appears in two splits. Adjacent frames differ by ~30 ms of probe motion; a
frame-level split scores the model on images it has effectively memorised and
inflates accuracy by roughly 10–20 points, uniformly, so the inflation never
shows up as noise.

Report **balanced accuracy, macro-F1, AUPRC, MCC** — not plain accuracy, which
sits in the low 80s for a model that always predicts "non-standard."

Two video-level metrics closer to the actual clinical use:

- `top1_frame_precision` — take the single highest-scoring frame per video;
  is it truly a standard plane? This is literally the workflow: the tool
  proposes one frame for the clinician to measure AoP/HSD on.
- `video_detection_rate` and `video_false_alarm_rate` — missing a case
  entirely is a different failure from mislabelling frames within a case.

Thresholds are chosen on validation and frozen before test. Confidence
intervals come from bootstrapping **videos**, not frames. Model comparisons use
a **paired** video bootstrap on the same resample, which is far more powerful
than checking whether two independent CIs overlap — and "the CIs overlap" is
not a valid way to conclude no difference.

Three seeds minimum on the headline comparisons; report mean ± std.

## What will most likely go wrong

**Shortcut learning is the top risk.** Three hospitals and multiple scanner
models are represented. If standard-plane frames aren't uniformly distributed
across centres, a model can score well by recognising the scanner — depth
markers, UI overlay, sector fan geometry, speckle statistics — and never look
at the symphysis. `src/gradcam.py` turns this into a number: the fraction of
Grad-CAM mass falling inside the PS/FH segmentation masks, divided by what an
equal-area random mask would collect. A ratio near 1.0 means no anatomical
preference. Compute it *before* you trust any accuracy figure. If it comes out
low: crop the UI region, and run a leave-one-centre-out split.

**The official IUGC test set has no public labels.** It cannot be your test
set. Build your own held-out split from train+validation (the code does this)
and state it plainly in the paper — this is standard practice for a closed
challenge and reviewers accept it without complaint.

**Class imbalance drifts between conditions.** The curated sparse arm is
positive-enriched and the dense arm is not, so part of any raw difference is
just a difference in class prior. Identical inverse-frequency weighting
everywhere is what removes it; do not tune the weighting per arm.

**A negative result is a good result here.** If dense loses under Protocol B,
or the LSTM ties the smoothed baseline, that is a cleaner and more publishable
finding than a small positive difference — because it is exactly the
measurement prior work failed to make. Do not go looking for a configuration
that reverses it.
