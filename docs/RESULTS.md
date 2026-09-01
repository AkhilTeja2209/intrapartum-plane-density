# Results

All numbers: ResNet-18, official IUGC split, scored on the identical complete
test set of **8,665 frames / 300 videos** (positive rate 0.556). Thresholds
chosen on validation and frozen before test. Confidence intervals are 95% from
a bootstrap over **videos**, not frames.

**Seed 0 unless stated.** Repeat seeds for the headline conditions are running;
until they land, everything below is a direction rather than a claim. The
reason is in §4 and it is not boilerplate — the spread within a single arm is
as large as the gaps between arms.

---

## 1. Arm 1 — the density curve

Sampling is applied to the training split only.

| condition | train frames | train videos | pos-rate | macro-F1 | bal-acc | AUPRC | top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sparse_k1` | 434 | 434 | 0.613 | **0.6872** | 0.6945 | 0.8373 | 0.8833 |
| `sparse_k2` | 868 | 434 | 0.613 | 0.5880 | 0.6252 | 0.8219 | 0.8633 |
| `sparse_k5` | 2,170 | 434 | 0.613 | 0.6431 | 0.6630 | 0.8373 | 0.8733 |
| `sparse_k10` | 4,339 | 434 | 0.613 | 0.6082 | 0.6392 | 0.8278 | 0.9033 |
| `sparse_k20` | 8,638 | 434 | 0.616 | 0.6386 | 0.6582 | 0.8326 | 0.8900 |
| `dense_stride8` | 6,840 | 434 | 0.414 | 0.6365 | 0.6567 | 0.7915 | 0.8633 |
| `dense_stride4` | 13,582 | 434 | 0.416 | 0.5058 | 0.5729 | 0.7784 | 0.8133 |
| `dense_stride2` | 27,062 | 434 | 0.417 | 0.6258 | 0.6499 | 0.7893 | 0.8667 |
| `dense_all` | 53,996 | 434 | 0.417 | 0.5785 | 0.6173 | 0.7689 | 0.7733 |

**No density effect is visible.** The sparse arm spans 0.588–0.687 over a
20&times; range of training frames with no monotone trend; the dense arm spans
0.506–0.637 over an 8&times; range, also with no trend. The best single result
in the whole study comes from **434 training frames** — one frame per video.

Note the dense arm does not beat the sparse arm even in **Protocol A**, which is
the comparison prior work reports as a win for video data. On this corpus, with
the confound removed by construction, it is not a win at all.

## 2. Arm 1 — Protocol B, matched budget

The contribution. Each dense arm is capped to the exact frame count of its
sparse counterpart; the prior-matched arms additionally hold the class balance
at the sparse arm's 0.613 (see `ROADMAP.md` R9).

| comparison | frames | sparse | dense | Δ |
|---|---:|---:|---:|---:|
| budget-matched, k5 | 2,170 | 0.6431 | 0.5370 | **−0.106** |
| budget-matched, k20 | 8,638 | 0.6386 | 0.5850 | **−0.054** |
| prior-matched, k5 | 2,170 | 0.6431 | 0.4751 | **−0.168** |
| prior-matched, k20 | 8,638 | 0.6386 | 0.5845 | **−0.054** |

**Dense loses all four**, and holding the class prior fixed does not rescue it —
it makes the k5 case worse. Reading this against the protocol's 2&times;2: dense
loses under B, so any advantage dense might have had in A would have been sample
count, not density. Here dense does not win A either, so the honest summary is
that **frame density buys nothing on this corpus, and concentrating a fixed
budget into fewer videos actively costs.**

The mechanism is visible in the data: `dense_matched_k5` draws its 2,170 frames
from roughly 20–25 videos, against 434 for `sparse_k5`. That is a statement
about **patient diversity**, not about frames.

## 3. Arm 2 — temporal modelling

All on `dense_all`, so the only difference is the head and the training-window
construction.

| model | macro-F1 | 95% CI | bal-acc | AUPRC | top-1 |
|---|---:|---|---:|---:|---:|
| frame-wise, raw | 0.5785 | [0.5479, 0.6069] | 0.6173 | 0.7689 | 0.7733 |
| frame-wise, **smoothed** (the protocol's baseline) | 0.5453 | — | — | — | — |
| BiLSTM, splicing off (ablation) | 0.6496 | [0.6243, 0.6760] | 0.6546 | 0.7831 | 0.8933 |
| BiLSTM, **splicing on** (`splice_p` 0.5) | **0.6695** | [0.6416, 0.6981] | 0.6808 | 0.7527 | 0.8000 |

Two findings, and the second undercuts the first in a way worth stating plainly.

**The temporal arm beats the frame-wise arm, including the smoothed baseline.**
0.6695 against 0.5453 is a 0.124 gap, and the temporal CI does not overlap the
frame-wise one. `PROTOCOL.md` predicted the opposite — that a tuned smoother
would match or beat an LSTM, making the temporal gain a post-processing
artefact. On this data it does not.

**But splicing is not what produced it.** The unspliced ablation reaches 0.6496
against the spliced 0.6695, and the intervals overlap heavily. The unspliced
model *cannot* have learned transitions — its training windows contain none —
so whatever the temporal head is contributing, it is not transition modelling.
The most likely explanation is the sliding-window logit averaging at inference,
which is itself a smoother, combined with clip-level context. That makes the
comparison against the frame-wise smoothed baseline less clean than it looks,
because the two arms are being smoothed by different mechanisms with different
tuning.

So splicing did its job — it made Arm 2 *runnable* on a split with no
transitions — but it did not turn out to be load-bearing for the result. Say
that in the paper rather than presenting splicing as the reason the LSTM wins.

## 4. Why none of this is a claim yet

Within the sparse arm alone, macro-F1 ranges 0.588–0.687 across conditions that
differ only in how many frames were drawn. That 0.099 spread is **larger than
three of the four matched-budget deficits** in §2. `dense_stride4` at 0.506 sits
0.13 below `dense_stride8` at 0.637 despite having twice the data.

That is the signature of run-to-run variance dominating the effect being
measured. The protocol calls for three seeds and a **paired** video bootstrap on
the same resample, and that is what would separate the two. Repeat seeds for the
headline conditions are in progress; the tables above will be revised, and the
direction may not survive.

Two things are already solid enough to state:

* Absolute performance is far below the 0.83–0.96 the protocol anticipated from
  simulation. That gap is the train/test regime shift (`docs/DATA_AUDIT.md` S2):
  the model trains on trimmed clips where the plane is held from first frame to
  last, and is tested on sweeps that move in and out of plane. Simulation
  assumed one population.
* Post-hoc smoothing **hurt in 12 of 13 conditions** (the exception is
  `sparse_k5`, 0.6431 → 0.6689). The tuner selects on validation, where runs are
  long — median 33 frames — but test runs are short, median 12. A filter tuned
  on the wrong run-length distribution over-smooths. This is a real
  methodological trap for anyone using the smoothed baseline the protocol
  specifies, and it is a consequence of val and test not being interchangeable.

## 5. What has not been run

* Seeds 1 and 2 (in progress), and the paired video bootstrap between arms.
* The Grad-CAM anatomical-attention check (`ROADMAP.md` R6). Until it runs, none
  of the above is protected against the model scoring well by recognising
  scanner or centre rather than anatomy — and with three hospitals in the
  corpus that risk is live.
* GRU vs LSTM, causal vs bidirectional, warm-start ablations.
* AoP/HSD downstream error (`ROADMAP.md` R7.1), the addition most likely to make
  the result mean something clinically.
