# Roadmap

Project outline, work breakdown, and recommended changes.
Companion documents: [`PROTOCOL.md`](PROTOCOL.md) (experimental design),
[`RUNBOOK.md`](RUNBOOK.md) (step-by-step execution),
[`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md) (what the data actually contains).

---

## 1. The project in one page

**Claim under test.** The literature reports that models trained on dense video
frames outperform models trained on curated still-image datasets, and
interprets this as evidence that *video data is richer*. That inference is
unsupported, because the dense arm also has far more training samples. Nobody
has separated the two.

**Design.** Build both arms from a single corpus — the IUGC 2024 intrapartum
transperineal ultrasound dataset — by varying only how many frames are drawn
per video. Probe, anatomy, scanners, annotators and label definition are then
constant by construction.

| Arm | Question | Manipulation | Baseline it must beat |
|---|---|---|---|
| 1A | Does frame density help? | k frames/video, k = 1 → all | itself at k=1 |
| 1B | **Is the gain density or sample count?** | dense arm capped to the sparse arm's exact frame budget | the matched sparse arm |
| 2 | Does temporal modelling help? | BiLSTM head on a frozen protocol | frame-wise CNN **+ tuned post-hoc smoothing** |

**1B is the contribution.** Prior work measures 1A and reports it as though it
were 1B. **Arm 2's baseline choice is the second contribution**: standard planes
occur in contiguous runs, so most exploitable temporal signal is label
autocorrelation, which a zero-parameter smoother recovers for free. Benchmarking
an LSTM against an *unsmoothed* CNN manufactures a temporal gain that is really
a post-processing artefact.

Negative results are wins here. "Density bought nothing once budget was matched"
and "the LSTM tied a moving average" are both cleaner findings than a small
positive effect, because they are the measurements the field skipped.

---

## 2. Where the project stands

| Stage | State |
|---|---|
| Pipeline code (15 modules, ~3,200 lines) | Written, coherent, well-documented |
| Dataset downloaded and unpacked | Done (774 videos) |
| Frame extraction | Done — 65,531 frames on disk |
| Frame index | **Broken** — 0 of 44,751 train frames carry a label |
| Splits / budgets | Not run (blocked on the index) |
| Any training run | Not started |
| Results, figures, Grad-CAM | Not started |

The code is in good shape. What is broken is the join between frames and
labels, and it is broken in a way that would not crash — it would train an
all-negative or empty classifier and report a number. Details and fixes:
[`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md).

---

## 3. Work breakdown

### Phase 0 — Unblock the index  *(hours, do first)*

- [ ] **B1** Handle the `ALL` sentinel in `parse_index_list`, expanding against `frame_count`
- [ ] **B2** Stop `unpack_dataset.py` producing `X__X` stems; normalise on join
- [ ] **B3** Reconcile `manifest.csv` against `index.csv`; account for the 71 lost train videos
- [ ] Turn the join-rate *log line* into an **assertion**: `assert join_rate == 1.0`
- [ ] Re-run `build_index`; confirm 434 train videos labelled and the pos-rate is plausible
- [ ] Extend `src/smoke_test.py` with a synthetic fixture using `ALL`/`NONE` sentinels and a doubled stem, so both bugs stay fixed

**Exit criterion:** zero frames with `label == -1` outside the intended set, and
a printed per-split video/frame/pos-rate table you have actually read.

### Phase 1 — Rebuild the design around the real data  *(days)*

- [ ] **R1** Adopt the official test split (labels are public — see audit S1)
- [ ] **R2** Decide and document the train weak-label strategy (audit S2)
- [ ] **R3** Recalibrate strata, weighting and prose to a 0.556 positive rate (audit S3)
- [ ] Rewrite the affected passages of `PROTOCOL.md` — it currently asserts three things about the data that are false
- [ ] Re-run `splits.py` and `make_budgets.py`; record the actual Protocol B budgets

### Phase 2 — Arm 1, the density curve  *(1–2 weeks of GPU)*

- [ ] Two extremes first: `sparse_k1` and `dense_all`, seed 0 — sanity, not science
- [ ] Grad-CAM anatomical-attention ratio **before** trusting any accuracy figure (R6)
- [ ] Fill the curve: k ∈ {2, 5, 10, 20}, strides {8, 4, 2}
- [ ] Protocol B: `dense_matched_k5`, `dense_matched_k20`
- [ ] Seeds 1 and 2 on the headline conditions
- [ ] Paired video bootstrap between arms

### Phase 3 — Arm 2, temporal modelling  *(1 week of GPU)*

- [ ] Tune the smoothing baseline on validation with the **same budget** the LSTM gets
- [ ] BiLSTM arm at the best density from Phase 2
- [ ] Ablations: GRU vs LSTM, causal vs bidirectional, ImageNet vs warm-start
- [ ] Report LSTM vs *smoothed* CNN as the headline comparison

### Phase 4 — Write-up

- [ ] Tables, density curve, paired-bootstrap significance
- [ ] Leave-one-centre-out result if the Grad-CAM check flagged shortcut learning
- [ ] Limitations: single corpus, weak train labels, 40-video validation set

---

## 4. Recommended changes

### R1 — Use the official test split. It has labels.  *(high impact, low cost)*

`PROTOCOL.md` builds a held-out set from train+val because it believes the test
labels are withheld. They are not: 300 test videos with frame-level annotation
ship in this release (audit S1).

Switch to **train → train, val → validation/threshold-tuning, test → test**.
You gain a test set untouched by any design decision, comparability with the
IUGC 2024 leaderboard, and one fewer paragraph of methodological apology. You
also avoid a subtle trap in the current design: a carved-from-train+val test set
would be dominated by long, *video-level-labelled* train videos, so you would be
scoring frame-level predictions against whole-video labels and calling it test
accuracy.

Caveat to state plainly: 40 validation videos is thin for threshold selection
and smoother tuning. Prefer **grouped 5-fold cross-validation over train** for
tuning, with val folded in, and touch test exactly once per condition.

### R2 — Confront the weak-label problem head-on. It is a feature.  *(high impact)*

Train labels are one-per-video; val and test are per-frame (audit S2). The
current protocol implicitly treats train labels as frame-level, which is wrong
and would quietly cap performance while making the density curve
uninterpretable — at k=1 you draw one frame carrying a bag label, at k=all you
draw 778 frames all carrying the *same* bag label.

Three options, in order of my preference:

1. **Multiple-instance learning.** Treat each train video as a bag: a positive
   bag contains ≥1 standard plane, a negative bag contains none. Train with an
   attention-MIL or noisy-OR pooled head. This is the *correct* model of the
   label generating process, and it is a genuine methodological contribution on
   top of the density question.
2. **Frame-level with noisy positives**, plus co-teaching or a small-loss filter
   to down-weight frames in positive bags the model is confident are
   non-standard.
3. **Naive frame-level** (the current implicit choice), reported as a baseline
   for options 1–2 rather than as the main result.

Whichever you pick, **the density curve must be re-interpreted**: with bag
labels, sampling more frames per video adds *label noise* as well as data. That
is itself an interesting result — say so rather than hiding it.

I would also add a condition the current design lacks: **train on val's
frame-level labels only** (2,870 clean frames from 40 videos) and compare
against 44,751 noisy frames from 434 videos. That is a clean label-quality-
versus-quantity axis, it costs almost no GPU time, and it sharpens the same
argument the paper is already making.

### R3 — Recalibrate everything written against a 17% prior  *(low cost, do now)*

The real positive rate is 0.556 (audit S3). `stratum()` in `src/splits.py`
buckets at 0 / <0.10 / <0.30 / ≥0.30 and would place nearly every video in the
top bucket, so stratification silently stops doing anything. Rebucket on
quartiles of the observed distribution. Inverse-frequency weighting is now close
to a no-op — keep it for consistency across conditions, but stop citing
imbalance as a headline difficulty.

### R4 — Make the pipeline fail loudly  *(low cost, high leverage)*

Every bug found in this audit is silent. The pipeline logs a join rate and
carries on; it drops 71 videos and carries on. Convert the three most dangerous
log lines into assertions with `--force` escapes:

- join rate must be 1.0 for train and val
- manifest frame count must equal index frame count
- per video, positives and negatives must partition the frames exactly

Add a `python -m src.validate` target that runs every invariant against the real
index in under a minute, and run it after each regeneration.

### R5 — Version the artefacts, not just the code  *(low cost)*

`index.csv`, `splits.json` and the budgets in `configs/` are *inputs* to every
number in the paper, and they will change as the fixes above land. Write a
manifest hash (git SHA + SHA-256 of `index.csv` and `splits.json`) into every
results file, and have `analyze.py` refuse to combine runs whose hashes differ.
This is what stops you comparing a pre-fix `sparse_k1` against a post-fix
`dense_all` three weeks from now.

### R6 — Run the Grad-CAM check before, not after, the experiments  *(reordering)*

`PROTOCOL.md` already identifies shortcut learning across three hospitals as the
top risk, and `src/gradcam.py` already quantifies it. But the run order in
`README.md` puts it last. If the attention ratio comes back near 1.0, the model
is reading scanner UI rather than anatomy and **every density number is
measuring the wrong thing** — you want to know that after the first two runs,
not after thirty. The `seg/` masks it needs are already on disk (audit S4) and
nothing reads them yet.

Prepare the leave-one-centre-out split now as well, so it is ready if needed.

### R7 — Additions worth the effort, ranked

1. **Report AoP/HSD downstream error.** `landmark.json` ships ground-truth angle
   of progression and head–symphysis distance. Take each model's top-1 proposed
   frame and report the measurement error a clinician would inherit. This turns
   "macro-F1 0.91" into a number an obstetrician can act on, and it is the
   single strongest addition available — the data is already there.
2. **A calibration section.** Reliability diagrams and expected calibration error
   per condition. Smoothing and temporal models change calibration sharply, and
   a frame proposer whose confidence is meaningless is not deployable.
3. **Cost axis on the density curve.** Plot macro-F1 against annotation cost and
   GPU cost, not only against k. "Density helps, and here is what it costs" is
   the practitioner-facing version of the result.
4. **FETAL_PLANES_DB transfer arm** — as `PROTOCOL.md` already scopes it, purely
   as a transfer source and external sanity check, never as a second condition.
5. **Inference latency** for the causal LSTM vs the smoothed CNN. The causal
   model is the only one deployable during live scanning; if it cannot keep up
   with the frame rate, say so.

### R8 — Repository and reproducibility hygiene  *(low cost)*

- [x] `.gitignore` covering `data/`, checkpoints and the 1.1 GB zip
- [ ] Pin exact versions in `requirements.txt` (`>=` will not reproduce in 2027)
- [ ] CI running `python -m src.smoke_test --synthetic` on every push — the smoke
      test exists and nothing runs it automatically
- [ ] `CITATION.cff`, and dataset attribution under CC-BY-4.0
- [ ] Log seed, git SHA and full resolved config into every results file
- [ ] Move `inspect_labels.py`, `inspect_zip.py`, `setup_project.py` and
      `unpack_dataset.py` into `tools/` — they are one-off scaffolding and
      currently sit alongside the pipeline as though they were part of it

---

## 5. Risk register

| Risk | Likelihood | Effect if ignored | Mitigation |
|---|---|---|---|
| Silent label mis-join | **Occurred** | Every number wrong, no error raised | R4 assertions |
| Shortcut learning on scanner identity | Medium | Density result measures the wrong thing | R6, run early |
| Weak train labels read as frame labels | **Occurred in design** | Density curve uninterpretable | R2 |
| 40-video val set overfits threshold/smoother | Medium | Optimistic test numbers | R1 cross-validation |
| GPU budget overrun (30–40 GPU-h) | Medium | Unfinished grid | `RUNBOOK.md` Part 11 cut order |
| Comparing runs across pipeline versions | High | Irreproducible headline table | R5 hashes |
