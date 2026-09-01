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

## S2 — Train labels are **video-level**; val/test are **frame-level**

This is the deepest structural fact about the dataset and it is not reflected
anywhere in the current protocol.

| split | annotation granularity | videos | mean frames/video |
|---|---|---:|---:|
| train | one label for the whole video (`ALL`/`NONE`) | 434 | 124 (9–778) |
| val | per-frame index lists | 40 | 72 (41–101) |
| test | per-frame index lists | 300 | 29 (20–70) |

A "standard-plane" train video is one *containing* a standard plane; its 124
frames are not all standard planes in the ordinary sense. Training on them as
frame labels is **learning from noisy positive bags**, not clean supervision.

`build_index` now measures and reports this rather than leaving it to be
inferred — it counts videos whose frames all carry one label:

```
train        434 / 434 videos single-label  -> video-level (weak) labels
val            3 /  40 videos single-label  -> frame-level labels
test          37 / 300 videos single-label  -> frame-level labels
```

434 of 434 is the signature of whole-video annotation. Consequences, in order of
severity, are worked through in `ROADMAP.md` R2.

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
