# Data audit — DatasetV3 as actually shipped

Findings from inspecting the unpacked `data/DatasetV3` tree and the generated
`data/index.csv` on 2026-09-02. Three of these contradict assumptions written
into `README.md` and `PROTOCOL.md`, so read this before trusting either.

## Summary of the index as it currently stands

| | videos | frames | labelled | pos-rate |
|---|---:|---:|---:|---:|
| train | 434 | 44,751 | **0** | — |
| val | 40 | 2,870 | 2,870 | 0.556 |
| test | 300 | 8,665 | 8,665 | 0.556 |

**Zero labelled training frames.** Nothing can be trained until this is fixed.

---

## B1 — `parse_index_list` does not handle the `ALL` sentinel  *(blocker)*

`train/cls/class_label.csv` uses two sentinel strings, documented in the
dataset's own `README_EN.md`:

> `pos_index`: … `"NONE"` means no standard plane, `"ALL"` means all frames are
> standard planes, or a specific index list

Distribution over the 434 train videos:

| `pos_index` | `neg_index` | videos |
|---|---|---:|
| `ALL` | `NONE` | 266 |
| `NONE` | `ALL` | 168 |

`parse_index_list` in `src/build_index.py` returns the empty set for `"NONE"`
(correct) and *also* for `"ALL"` — the string is not a Python literal, so it
falls through to the regex branch, which finds no digits and returns `set()`.
Every train video therefore contributes zero labelled frames.

The fix is a single case, but it must expand against `frame_count`:

```python
if s.upper() == "ALL":
    if frame_count is None:
        raise ValueError("'ALL' needs frame_count to expand")
    return set(range(frame_count))
```

Note the asymmetry this creates and guard it: with `pos_index=ALL` and
`neg_index=NONE`, the two sets must partition the video exactly. Assert
`len(pos | neg) == frame_count` and `not (pos & neg)` per video, and fail
loudly rather than silently dropping frames.

## B2 — 266 of 434 train video filenames do not join  *(blocker)*

The unpack step produced doubled stems on disk for exactly the 266
standard-plane videos:

| source | stem |
|---|---|
| `class_label.csv` | `20190909T155747I1` |
| on disk / `manifest.csv` | `20190909T155747I1__20190909T155747I1` |

168 stems join; 434 − 168 = 266 do not — the same 266 that carry
`pos_index=ALL`. So B1 and B2 independently destroy *every positive training
video*, and fixing only one of them still yields an all-negative training set
that trains to a degenerate classifier without erroring.

Fix in `unpack_dataset.py` (stop generating `X__X`), then normalise stems on
join with a `stem.split("__")[0] if stem == doubled else stem` rule, and
**assert the join rate is 1.0** rather than logging it.

## B3 — 71 train videos and 9,245 frames vanish between manifest and index

`data/frames/manifest.csv` records 774 videos / 65,531 frames.
`data/index.csv` contains 703 videos / 56,286 frames. The 71 missing videos are
all in the train split. Cause not yet established — likely the same stem
mismatch, or extraction failures swallowed by a `try/except`. Add a reconcile
assertion between manifest and index.

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

Consequences, in order of severity, are worked through in `ROADMAP.md` R2.

## S3 — The class prior is ~56% positive, not ~17%

`PROTOCOL.md` assumes a ~17% standard-plane rate and warns that "plain accuracy
sits in the low 80s for a model that always predicts non-standard." On the real
frame-level splits the positive rate is **0.556**. A majority-class classifier
scores ~56%, not ~83%.

The imbalance-aware metric set is still the right choice, but the specific
numbers, the `stratum()` buckets in `src/splits.py` (which top out at
`pos_rate >= 0.30` and would collapse nearly every video into one bucket), and
the inverse-frequency weighting rationale all need rewriting against 0.556.

## S4 — Unused assets: segmentation masks and AoP/HSD landmarks

`seg/` ships PS/fetal-head masks, and `landmark.json` ships `ps_points`,
`hsd_point`, `aop_tangency` and ground-truth `aop` / `hsd` values (coordinates
are `[y, x]`, strings, origin top-left). `src/gradcam.py` needs exactly these
masks for its anatomical-attention ratio, and nothing in the pipeline reads
them yet.
