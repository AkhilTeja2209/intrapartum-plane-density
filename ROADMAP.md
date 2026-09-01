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
| Frame extraction | Done — 65,531 frames across all 774 videos |
| Frame index | Done — 65,531 frames, join rate 1.0000, 0 unlabelled |
| Splits | Done — official scheme, leakage assertions pass |
| Budgets | Not run |
| Any training run | Not started |
| Results, figures, Grad-CAM | Not started |

Phase 0 is complete and Phase 1's three design questions are settled: R1 (use
the official split), R2 (train frame-wise), R3 (data-derived strata). One
decision remains open and it blocks Phase 3 only — how to give the temporal arm
the label transitions the training split does not contain. Evidence for all of
it: [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md).

The index as rebuilt:

| split | videos | frames | pos-rate | annotation |
|---|---:|---:|---:|---|
| train | 434 | 53,996 | 0.418 | trimmed clips — **0.00** label transitions/video |
| val | 40 | 2,870 | 0.556 | sweeps — 0.93 transitions/video |
| test | 300 | 8,665 | 0.556 | sweeps — 1.35 transitions/video |

---

## 3. Work breakdown

### Phase 0 — Unblock the index  *(done)*

- [x] **B1** Resolve the `ALL` sentinel against `frame_count`, and require sentinel rows to partition the video exactly
- [x] **B2** Collapse the doubled `X__X` stems shipped in the Zenodo zip, on the join key only
- [x] **B3** Fix `cv2.imwrite` silently failing on non-ASCII paths; all 71 lost videos recovered
- [x] **B4** Verify duplicate label files agree instead of keeping whichever was scanned first
- [x] Join rate, empty folders and manifest/disk agreement are now **assertions**, escapable with `--allow-unlabelled` / `--allow-empty-dirs`
- [x] `build_index` reports annotation granularity per split, so S2 is visible at build time
- [x] `src/smoke_test.py` stage 0 guards both defects (12 assertions, including the legitimate `__` that must survive)

**Result:** 65,531 frames / 774 videos, join rate 1.0000 on all three splits,
zero unlabelled. Full synthetic smoke test passes end to end.

### Phase 1 — Rebuild the design around the real data  *(days)*

- [x] **R1** Adopt the official split — `splits.py --scheme official` is the default; `regrouped` kept for sensitivity analysis
- [x] **R2** Train weak-label strategy decided: **frame-wise supervision**, no MIL. The labels are literal; what is missing is label *transitions* in train
- [x] **R3** `make_strata()` derives buckets from the data instead of assuming a 17% prior
- [x] Protocol B budgets computed against the real split and written to `configs/default.yaml` (2,170 and 8,638)
- [ ] **R9** Add the prior-matched dense arm — the class prior moves with the density axis (0.613 sparse vs 0.415 dense)
- [ ] **R10** Drop or redefine `sparse_k1_curated`; it is now a duplicate of `sparse_k1`
- [ ] Choose the Arm 2 remedy from R2 (splice / train-on-val / report-and-stop) — blocks Phase 3, not Phase 2
- [ ] Rewrite the affected passages of `PROTOCOL.md` — it still asserts three things about the data that are false
- [ ] Run `make_budgets.py` and record the actual Protocol B budgets

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

### R1 — DONE: use the official split  *(implemented in `splits.py`)*

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

There is a second, stronger reason that only emerged from R2: the two regimes
are not interchangeable. Train is 434 trimmed clips with zero label
transitions; val and test are sweeps. Pooling and redrawing puts trimmed clips
into the test set, where they are trivially easy — one stable anatomy, one
label, no transition to get wrong — and the reported score becomes a
seed-dependent weighted average over two populations. The regrouped scheme on
seed 0 yields a test set of 95 trimmed clips and 60 sweeps, which is not a
measurement of anything in particular.

`splits.py --scheme official` is now the default. The old behaviour survives as
`--scheme regrouped` for sensitivity analysis, and warns when it mixes regimes.

Caveat to state plainly: 40 validation videos is thin for threshold selection
and smoother tuning. Prefer **grouped 5-fold cross-validation over train** for
tuning, with val folded in, and touch test exactly once per condition. That is
not yet implemented.

### R2 — DECIDED: train frame-wise. The labels are real; the *transitions* are missing.  *(resolved 2026-09-02)*

I previously ranked multiple-instance learning first here, on the reading that
whole-video `ALL`/`NONE` labels were bag labels over noisy positives. **The
dataset's own segmentation annotations disprove that reading**, so the decision
goes the other way.

**Evidence** (full working in audit S2): segmentation masks exist for exactly
the 266 `ALL` videos and no others; they sit at ~10 frames spanning 10–98% of
each clip; all 2,575 annotated frames yield a measurable angle of progression;
and AoP drifts by a median of 5.2° within a video. You cannot segment the pubic
symphysis and fetal head, or measure AoP, on a frame that is not a standard
plane. The annotators found the plane held across the whole clip because these
are **trimmed standard-plane clips**, not sweeps that happen to contain one.

**Decision: option 3 — ordinary frame-wise supervision.** No MIL, no
noisy-label correction. MIL would model a noise process that is not there and
would discard genuine per-frame labels in exchange for one bag label per video;
co-teaching would down-weight frames that are correctly labelled. Both would
cost accuracy and add a confound to a study whose independent variable is
sampling density.

**What replaces the weak-label problem is worse, and it lands on Arm 2.** No
training video contains a single label transition (0.00 per video, against 0.93
in val and 1.35 in test). A BiLSTM trained there reaches zero training loss by
ignoring time and emitting one constant per clip — it never sees the event a
temporal model exists to model. The smoothed-CNN baseline meanwhile has its
window and Viterbi `p_stay` tuned on validation, which does contain
transitions. Run as specified, Arm 2 would measure the split rather than the
architecture, and the LSTM would lose for the wrong reason.

Three ways forward, in order of preference:

1. **Splice transitions into training.** Concatenate a positive clip with a
   negative clip from the *same* video's neighbours to synthesise the
   into-plane/out-of-plane boundary. This is honest augmentation — it mimics
   the probe moving off plane, which is exactly what a sweep does — and it is
   the only option that gives the temporal arm the signal it needs while
   keeping the official split. Report the splice rate as a hyperparameter.
2. **Train the temporal arm on val's sweeps** (37 mixed videos, median run 33
   frames), tune on a held-out slice of it, and test on test. Honest, but 37
   videos is very thin and the frame-wise and temporal arms would then be
   trained on different data — which breaks the "one protocol, one variable"
   discipline the rest of the study depends on.
3. **Report the negative result as a property of the corpus.** "The IUGC train
   split cannot teach temporal transitions, so no temporal architecture can be
   evaluated fairly on it" is a true and useful statement, and cheap. It is
   weaker than (1) because it forecloses the measurement rather than making it.

Arm 1 is unaffected in mechanism, but its *interpretation* shifts: within a
positive training video every frame shares one label and the anatomy is
near-static, so raising density adds redundant views of a single plane rather
than label diversity. The density curve is a **redundancy curve** — say so in
the paper, because a reviewer who works out the trimming will otherwise
conclude you missed it.

The extra condition proposed earlier — train on val's frame-level labels only —
is still worth running, and now for a sharper reason than label quality: it is
the only condition in which the frame-wise arm sees transitions at all.

### R3 — PARTLY DONE: recalibrate against the real prior

The real positive rate is 0.442 pooled, 0.556 on test (audit S3). The old
`stratum()` in `src/splits.py` cut at 0 / <0.10 / <0.30 / ≥0.30 — fixed points
chosen for an assumed ~17% prior — so nearly every video fell in the top bucket
and stratification silently stopped doing anything.

**Done:** `make_strata()` now derives buckets from the data. All-negative and
all-positive videos get their own strata (they are qualitatively different: a
video with no standard plane cannot contribute a positive to any fold) and the
interior is cut at quantiles, with under-populated strata merged into their
neighbour so the fold constraint stays satisfiable. On the real index this
yields `{none: 168, mixed0: 78, mixed1: 72, mixed2: 77, mixed3: 73, all: 306}`
instead of one bucket holding everything.

**Still to do:** inverse-frequency weighting is now close to a no-op — keep it
for consistency across conditions, but stop citing imbalance as a headline
difficulty, and rewrite the affected prose in `PROTOCOL.md`.

### R4 — Make the pipeline fail loudly  *(done for `build_index`; extend outward)*

Every defect in this audit was silent. The pipeline logged a join rate and
carried on; it dropped 71 videos and carried on; it resolved a documented
sentinel to nothing and carried on. Four invariants are now enforced in
`build_index`, each with an explicit escape hatch rather than a default-on
tolerance:

- join rate must be 1.0 per split (`--allow-unlabelled` to override)
- no extracted folder may be empty, and the manifest must match the disk
  (`--allow-empty-dirs`)
- sentinel rows must partition their video exactly, with no overlap
- duplicate label files must agree, not merely deduplicate

Still to do: apply the same treatment to `splits.py` and `run_experiment.py`,
and add a `python -m src.validate` target that re-checks every invariant against
a written index in under a minute, so the guarantees survive a hand-edited CSV.

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

### R9 — NEW: the class prior is confounded with the density axis  *(found while validating R1–R3; high impact)*

Building every condition against the real split exposes a step change in the
training class prior at exactly the sparse/dense boundary:

| condition | frames | pos-rate |
|---|---:|---:|
| `sparse_k1` … `sparse_k20` | 434 → 8,638 | **0.613** (flat) |
| `dense_stride8` … `dense_all` | 6,840 → 53,996 | **0.415** (flat) |
| `dense_matched_k5` | 2,305 | 0.548 |
| `dense_matched_k20` | 9,220 | 0.418 |

The cause is structural, not a bug. Per-video sampling weights every video
equally, so the prior is just 266/434 = 0.613. Dense sampling weights by video
length, and the trimmed positive clips are shorter (median 80 frames) than the
negative ones (median 96, mean 187), so the long negatives dominate. Density and
class prior therefore move together across the entire independent variable, and
Protocol B does not escape it either: `dense_matched_k5` sits at 0.548 against
`sparse_k5`'s 0.613.

`PROTOCOL.md` anticipates a version of this and answers it with identical
inverse-frequency weighting everywhere. That is not sufficient here. Identical
weighting rebalances *within* each condition; it does not make two conditions
with different priors comparable, because the reweighted losses are still
computed over different underlying distributions and the decision threshold that
weighting implies differs between them.

Recommended fix, cheap and decisive: add a **prior-matched** variant of the
dense arm that subsamples negatives until its positive rate equals the sparse
arm's 0.613, and report it beside the existing budget-matched arm. The 2×2 then
becomes a 2×3, and the three-way reading is clean:

| A (natural) | B (budget-matched) | B′ (budget- *and* prior-matched) | conclusion |
|---|---|---|---|
| dense wins | dense wins | dense wins | density carries real information |
| dense wins | dense wins | tie | the gain was the class prior, not density |
| dense wins | dense loses | — | the gain was sample count |

Without B′, a reviewer can attribute any dense-arm advantage to its different
class balance, and the paper has no answer.

### R10 — NEW: `sparse_k1_curated` no longer tests anything  *(drop it)*

The curated condition exists to mimic how a sonographer builds an image dataset:
freeze at the temporal centre of a standard-plane run, which yields a
positive-enriched set (~76% in the original simulation against ~17% for uniform).

That depends on runs existing. In the official train split there are none — every
positive video is one unbroken standard-plane clip (R2). Measured on the real
data, `sparse_k1_curated` and `sparse_k1` produce **identical class balance
(0.6129 both)** and share 44% of their exact frames; they differ only in which
arbitrary frame gets picked from a video where every frame carries the same
label.

Drop it from the grid, or redefine it against the sweep-structured val/test
videos where runs do exist. Keeping it as-is spends GPU time on a condition
whose defining property the data cannot express — and reporting it as a
"realism check" would misdescribe what was measured.

---

## 5. Risk register

| Risk | Likelihood | Effect if ignored | Mitigation |
|---|---|---|---|
| Silent label mis-join | **Occurred** | Every number wrong, no error raised | R4 assertions |
| Shortcut learning on scanner identity | Medium | Density result measures the wrong thing | R6, run early |
| Train split contains no label transitions | **Confirmed** | Arm 2 measures the split, not the architecture | R2 remedy, blocks Phase 3 |
| Density curve read as information gain when it is redundancy | High | Overclaimed conclusion | R2, state it in the paper |
| 40-video val set overfits threshold/smoother | Medium | Optimistic test numbers | R1 cross-validation |
| GPU budget overrun (30–40 GPU-h) | Medium | Unfinished grid | `RUNBOOK.md` Part 11 cut order |
| Comparing runs across pipeline versions | High | Irreproducible headline table | R5 hashes |
| Class prior moves with the density axis | **Confirmed** | Dense-arm gain attributable to class balance | R9 prior-matched arm |
