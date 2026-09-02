# Results

All numbers: ResNet-18, official IUGC split, scored on the identical complete
test set of **8,665 frames / 300 videos** (positive rate 0.556). Thresholds
chosen on validation and frozen before test. Confidence intervals are 95% from
a bootstrap over **videos**, not frames.

**§1–3 are seed 0. §4 repeats the headline conditions over three seeds, and it
overturns the single-seed reading — read it before quoting anything above.**

---

## 1. Arm 1 — the density curve  *(seed 0)*

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

**No density effect is visible** (and see §4 — the seed-to-seed spread on a
single condition reaches 0.23, so the ordering within this table carries little
information). The sparse arm spans 0.588–0.687 over a
20&times; range of training frames with no monotone trend; the dense arm spans
0.506–0.637 over an 8&times; range, also with no trend. The best single result
in the whole study comes from **434 training frames** — one frame per video.

Note the dense arm does not beat the sparse arm even in **Protocol A**, which is
the comparison prior work reports as a win for video data. On this corpus, with
the confound removed by construction, it is not a win at all.

## 2. Arm 1 — Protocol B, matched budget  *(seed 0; refuted in §4)*

The contribution. Each dense arm is capped to the exact frame count of its
sparse counterpart; the prior-matched arms additionally hold the class balance
at the sparse arm's 0.613 (see `ROADMAP.md` R9).

| comparison | frames | sparse | dense | Δ |
|---|---:|---:|---:|---:|
| budget-matched, k5 | 2,170 | 0.6431 | 0.5370 | **−0.106** |
| budget-matched, k20 | 8,638 | 0.6386 | 0.5850 | **−0.054** |
| prior-matched, k5 | 2,170 | 0.6431 | 0.4751 | **−0.168** |
| prior-matched, k20 | 8,638 | 0.6386 | 0.5845 | **−0.054** |

At seed 0 dense loses all four, which reads as a clean Protocol B result:
concentrating a fixed budget into fewer videos costs you, and the mechanism
looks visible in the data — `dense_matched_k5` draws its 2,170 frames from
roughly 20–25 videos against 434 for `sparse_k5`, so it would be a statement
about patient diversity rather than frames.

> **This does not replicate.** Two of these four comparisons were repeated over
> three seeds in §4 and both **flip sign**. Do not quote this table on its own.

## 3. Arm 2 — temporal modelling  *(seed 0 only)*

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

## 4. Three seeds, and what survives

The single-seed tables above looked like a result. They are not one.

| condition | mean | std | min | max | **range** |
|---|---:|---:|---:|---:|---:|
| `sparse_k1` | 0.6823 | 0.0529 | 0.6271 | 0.7325 | 0.105 |
| `sparse_k5` | 0.6383 | 0.0503 | 0.5858 | 0.6860 | 0.100 |
| `dense_matched_k5` | 0.5764 | 0.0343 | 0.5370 | 0.5990 | 0.062 |
| `dense_prior_matched_k5` | 0.6076 | **0.1193** | 0.4751 | 0.7065 | **0.231** |
| `dense_all` | 0.6102 | 0.0315 | 0.5785 | 0.6415 | 0.063 |

`dense_prior_matched_k5` moves by **0.23 macro-F1** between seeds on identical
data with an identical recipe. Nothing in §2 is larger than that.

### Protocol B does not replicate

| comparison | seed 0 | seed 1 | seed 2 | mean ± std | winner |
|---|---:|---:|---:|---|---|
| `dense_matched_k5` − `sparse_k5` | −0.106 | **+0.013** | −0.093 | −0.062 ± 0.065 | 2 sparse / 1 dense |
| `dense_prior_matched_k5` − `sparse_k5` | −0.168 | **+0.121** | −0.045 | −0.031 ± 0.145 | 2 sparse / 1 dense |

**Both comparisons flip sign across seeds, and both standard deviations are
larger than their means.** The seed-0 finding — "dense loses all four
matched-budget comparisons" — does not survive contact with two more seeds.
Protocol B is **unresolved** on this corpus at n=3.

### What that leaves

The honest headline is not about density. It is about measurability:

> On a 434-video corpus with this train/test regime shift, seed-to-seed
> variance reaches ±0.12 macro-F1 (range 0.23). That is larger than any density
> effect present. A single-run sparse-versus-dense comparison on a dataset this
> size cannot distinguish a real effect from a reseed.

That indicts the methodology this project set out to critique more directly
than a positive result would have. Prior work reports Protocol A from single
runs; here, single runs disagree with each other by more than the quantity being
measured.

Two weaker directional statements survive, both short of significance at n=3:

* `sparse_k1` has the highest mean (0.6823 ± 0.053) and `dense_matched_k5` the
  lowest (0.5764 ± 0.034) — a 0.106 gap at roughly 2.4 pooled standard
  deviations.
* More frames did not help: 434 frames (`sparse_k1`, 0.682) against 53,996
  (`dense_all`, 0.610), a gap of 0.072 at about 1.6 pooled standard deviations,
  in the *opposite* direction to the literature's claim.

Neither is a claim. The protocol's **paired video bootstrap on a common
resample** is far more powerful than comparing these independent means and is
the correct next test; comparing whether the CIs overlap is not a valid
substitute and is not what is being done here.

`dense_prior_matched_k5` deserves a note of its own: its std (0.119) is roughly
3&times; the other conditions'. Prior matching subsamples negatives to hit 0.613,
so each seed draws a different and smaller set of videos, adding variance on top
of training noise. If the prior-matched arm is kept, it needs more seeds than
the others, not the same number.

## 5. What has not been run

* The **paired video bootstrap** between arms on a common resample — the test
  the protocol specifies, and the one that would extract whatever signal exists
  from under the variance documented in §4.
* Seeds beyond 3, especially for the prior-matched arm. Arm 2 has one seed only,
  so §3's temporal result is exactly as provisional as §1's was.
* The Grad-CAM anatomical-attention check (`ROADMAP.md` R6). Until it runs, none
  of the above is protected against the model scoring well by recognising
  scanner or centre rather than anatomy — and with three hospitals in the
  corpus that risk is live.
* GRU vs LSTM, causal vs bidirectional, warm-start ablations.
* AoP/HSD downstream error (`ROADMAP.md` R7.1), the addition most likely to make
  the result mean something clinically.
