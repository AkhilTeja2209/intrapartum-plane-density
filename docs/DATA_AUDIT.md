# Data audit — DatasetV3 as actually shipped

Findings from inspecting the unpacked `data/DatasetV3` tree and the generated
`data/index.csv`, 2026-09-02.

**B1–B4 are pipeline defects and are fixed.** Every one of them was silent: no
exception, no non-zero exit, just wrong numbers. **S1–S4 are properties of the
dataset that contradict assumptions written into `README.md` and `PROTOCOL.md`**
— those are design decisions, not bugs, and they are still open. Read this
before trusting either document.

## Summary of the index

All three blocking defects below are **fixed**; the figures are the rebuilt
index. The original broken state is recorded in each section, because the
failure mode — not one of them raised an error — is the reusable lesson.

| | videos | frames | labelled | pos-rate |
|---|---:|---:|---:|---:|
| train | 434 | 53,996 | 53,996 | 0.418 |
| val | 40 | 2,870 | 2,870 | 0.556 |
| test | 300 | 8,665 | 8,665 | 0.556 |
| **total** | **774** | **65,531** | **65,531** | **0.442** |

Join rate is 1.0000 on every split. Before the fixes: 703 videos, 56,286
frames, and **zero labelled training frames**.

---

## B1 — `parse_index_list` did not handle the `ALL` sentinel  *(fixed)*

`train/cls/class_label.csv` uses two sentinel strings, documented in the
dataset's own `README_EN.md`:

> `pos_index`: … `"NONE"` means no standard plane, `"ALL"` means all frames are
> standard planes, or a specific index list

Distribution over the 434 train videos:

| `pos_index` | `neg_index` | videos |
|---|---|---:|
| `ALL` | `NONE` | 266 |
| `NONE` | `ALL` | 168 |

`parse_index_list` returned the empty set for `"NONE"` (correct) and *also* for
`"ALL"` — the string is not a Python literal, so it fell through to the regex
branch, which found no digits and returned `set()`. Every train video therefore
contributed zero labelled frames.

Fixed by resolving the sentinel against the row's `frame_count`. Three
guardrails came with it, because a sentinel that silently resolves to nothing is
exactly what caused the original bug:

- `ALL` without a `frame_count` column raises instead of returning `set()`
- for any sentinel row, `pos` and `neg` must be **disjoint** — a frame in both
  means the blanket sentinel contradicts an explicit list, and letting the
  sentinel win would discard the more specific annotation
- for any sentinel row, `pos | neg` must cover **exactly** `frame_count`

## B2 — 266 train filenames are doubled **in the published zip**  *(fixed)*

The train videos ship under doubled stems, and the label CSV does not use them:

| source | stem |
|---|---|
| `class_label.csv` | `20190909T155747I1` |
| `train/videos/` in the Zenodo zip | `20190909T155747I1__20190909T155747I1` |

168 stems joined; the other 266 did not — the same 266 carrying `pos_index=ALL`.
So B1 and B2 independently destroyed *every positive training video*, and fixing
only one of them still yielded an all-negative training set that would train to
a degenerate classifier without erroring.

This is an upstream packaging defect: the doubling is inside the distributed
archive, so no unpack or extract step can avoid it. It is reconciled at join
time by `canonical_video_id()`, which collapses a stem only when it is exactly
`H__H`. The exactness matters — legitimate filenames in this dataset use `__` as
an ordinary separator (`20190830T115644__B_产科_tmp_0`), and splitting on the
first occurrence would mangle them. `frame_path` keeps the real on-disk
directory name, so only the join key is rewritten and the loader still resolves.

## B3 — 71 videos extracted zero frames: `cv2.imwrite` and non-ASCII paths  *(fixed)*

`manifest.csv` recorded 774 videos / 65,531 frames while the disk held 703
videos / 56,286 frames. The 71 missing videos are exactly those whose filenames
contain Chinese characters (`..._B_产科_tmp_0`).

`cv2.imwrite()` hands the path to OpenCV's C++ layer, which on Windows encodes
it with the process ANSI codepage. For these paths the encoding fails, and
**imwrite reports that by returning `False`** — which nothing was checking. The
extractor decoded each video, counted the frames, wrote the count into the
manifest, and produced an empty folder. The videos looked extracted and vanished
from the study without a single error.

Fixed by encoding in memory with `cv2.imencode()` and writing through
`Path.write_bytes()`, so the path never reaches OpenCV. `build_index` now also
refuses to run when any extracted folder is empty or disagrees with the
manifest, gated behind `--allow-empty-dirs`.

## B4 — Duplicate label files were deduplicated without being compared  *(fixed)*

`cls_label.csv` and `*_info.csv` encode the same val/test labels twice — 11,535
overlapping `(video, frame)` pairs. The merge kept the first row seen, so a
genuine disagreement between the two files would have been resolved by
filesystem scan order. The copies do in fact agree everywhere, but that was
never checked. `build_index` now verifies agreement and raises on conflict.

---

## S1 — The official test labels **are** public in DatasetV3  *(design change)*

`PROTOCOL.md` states the IUGC test set ships without labels and that a held-out
set must be carved from train+val. That was true of the challenge release; it is
not true of this Zenodo deposit. `test/cls/cls_label.csv` contains full
frame-level `pos_index` / `neg_index` lists for all 300 test videos, and 8,665
test frames are already labelled in `index.csv`.

This is strictly good news and it changes the evaluation design — see
[`ROADMAP.md`](../ROADMAP.md) R1.

## S2 — Train is **trimmed clips**, val/test are **untrimmed sweeps**

An earlier version of this section read the whole-video `ALL`/`NONE` labels as
weak *bag* labels — "this video contains a standard plane somewhere" — and
concluded the training data was noisy positives. **That was wrong, and the
dataset's own segmentation annotations disprove it.**

`ALL` is meant literally. Evidence:

| test | result |
|---|---|
| Videos with PS/fetal-head segmentation | 266 — **exactly** the 266 `pos_index=ALL` videos, and none of the 168 negatives |
| Masks per positive video | 9.7 on average, spanning 10%→98% of the clip; 246/266 span more than 80% |
| Annotated frames yielding a measurable AoP | 2,575 / 2,575 |
| Within-video AoP spread | median **5.2°** (p90 14°) |

Segmenting the pubic symphysis and fetal head is only possible on a measurable
plane, and an angle of progression can only be computed once both are resolved.
Annotators produced both at frames spread across the entire clip, and the
measured anatomy barely moves. The positive training videos are **trimmed
standard-plane clips**: a held plane from first frame to last. Frame counts
agree — positives run 40–157 frames (median 80) while negatives run 9–778
(median 96).

So the training labels are genuine frame-level labels, coarsely encoded. There
is no bag-label problem to solve.

### The real consequence: no training video contains a label transition

| split | videos | transitions/video | videos that are entirely one class |
|---|---:|---:|---|
| train | 434 | **0.00** | 434 / 434 |
| val | 40 | 0.93 | 3 / 40 |
| test | 300 | 1.35 | 37 / 300 |

Run lengths in the mixed videos: val median 33 frames, test median 12.

Train is 266 clips of pure standard plane plus 168 clips of pure non-standard.
Val and test are sweeps that pass into and out of plane. Two consequences, and
the second is severe:

1. **Arm 1** is affected only in interpretation. Within a positive training
   video every frame carries the same label and the anatomy is near-static
   (5.2° of AoP drift), so raising sampling density adds *redundant views of one
   plane*, not label diversity. The density curve is still measurable; it is a
   redundancy curve, and the paper should call it that.

2. **Arm 2 cannot be run on this split as specified.** A BiLSTM trained where
   the label never changes can reach zero training loss by ignoring time and
   emitting one constant per clip — it never observes a transition, which is
   the only thing a temporal model has to contribute. The smoothed-CNN baseline,
   by contrast, has its window and Viterbi `p_stay` tuned on validation, which
   *does* contain transitions. Comparing them would measure the split, not the
   architecture. `ROADMAP.md` R2 carries the options.

`src/splits.py` computes and reports transitions per video for exactly this
reason, and warns when the training split contains none.

## S3 — The class prior is ~56% positive, not ~17%

`PROTOCOL.md` assumes a ~17% standard-plane rate and warns that "plain accuracy
sits in the low 80s for a model that always predicts non-standard." The rebuilt
index says otherwise:

| split | positive rate |
|---|---:|
| train (video-level labels) | 0.418 |
| val | 0.556 |
| test | 0.556 |
| pooled | 0.442 |

A majority-class classifier scores ~56% on test, not ~83%. Nothing here is
meaningfully imbalanced.

The imbalance-aware metric set is still the right choice, but the specific
numbers, the `stratum()` buckets in `src/splits.py` (which top out at
`pos_rate >= 0.30` and would collapse nearly every video into one bucket), and
the inverse-frequency weighting rationale all need rewriting against these
figures.

## S4 — Unused assets: segmentation masks and AoP/HSD landmarks

`seg/` ships PS/fetal-head masks, and `landmark.json` ships `ps_points`,
`hsd_point`, `aop_tangency` and ground-truth `aop` / `hsd` values (coordinates
are `[y, x]`, strings, origin top-left). `src/gradcam.py` needs exactly these
masks for its anatomical-attention ratio, and nothing in the pipeline reads
them yet.
