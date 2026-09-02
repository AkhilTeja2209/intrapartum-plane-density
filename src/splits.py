"""Split by VIDEO, never by frame.

This is the single most important file in the repo. Consecutive ultrasound
frames are near-duplicates: frame 340 and frame 341 differ by ~30 ms of probe
motion. If a random frame-level split puts 340 in train and 341 in test, the
model is scored on images it has effectively memorised, and reported accuracy
inflates by roughly 10-20 points. Every number in the study would be wrong in
the same direction, so the inflation would not even show up as noise.

Every split produced here is therefore grouped by video_id, and the grouping is
asserted rather than assumed.

Two schemes
-----------

`official` (default) uses the dataset's own train / val / test folders.

`regrouped` pools every labelled video and re-partitions with
StratifiedGroupKFold. It was the original default, on the belief that the
official test labels were withheld. They are not,
and pooling is actively harmful here -- for a reason specific to this dataset,
documented below.

Why the official split is not just acceptable but required
----------------------------------------------------------

The three folders are not three samples of one population. They are two
different kinds of recording:

    train   434 videos, 0.00 label transitions per video, 100% single-class
    val      40 videos, 0.92 transitions per video,  37/40 mixed
    test    300 videos, 1.35 transitions per video, 263/300 mixed

The 266 positive training videos are **trimmed standard-plane clips**: the
annotators segmented the pubic symphysis and fetal head at ~10 frames spread
across 10-98% of each clip, every one of those 2,575 frames yields a measurable
angle of progression, and AoP varies by a median of only 5.2 degrees within a
video. The clip is a held plane, start to finish. The 168 negative training
videos are the same idea inverted. Val and test are **untrimmed sweeps** that
pass into and out of plane.

Pooling the two regimes and redrawing would put trimmed clips into the test set,
where they are trivially easy -- one stable anatomy, one label, no transition to
get wrong -- and the reported score would become a weighted average over two
populations, with the weights set by the random seed. That is a worse
measurement than the distribution shift it was trying to avoid.

Training on curated clips and evaluating on live sweeps is also the honest
deployment story, so the shift belongs in the paper rather than being hidden.

What this costs the temporal arm
--------------------------------

A model trained only on the official train split never sees a single label
transition. A BiLSTM can drive its training loss to zero by ignoring time and
emitting one constant per clip, while the smoothed-CNN baseline gets its
transition prior tuned on validation, which does have transitions. Arm 2 as
specified would compare a temporal model that could not learn transitions
against a baseline tuned on them, and the LSTM would lose for reasons that have
nothing to do with temporal modelling. This module reports the transition
counts so the problem cannot be run past by accident. SplicedClipDataset in
src/datasets.py synthesises the missing transitions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .utils import get_logger, save_json

log = get_logger("splits")

SCHEMES = ("official", "regrouped")
OFFICIAL_ORDER = ("train", "val", "test")


def video_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per video: frame count, standard-plane fraction, and the
    temporal structure of its labels.

    `n_transitions` counts how often the label changes between consecutive
    frames. It is the number that distinguishes a trimmed clip (0) from a sweep
    (>=1), and it is what Arm 2 actually depends on.
    """
    df = df.sort_values(["video_id", "frame_idx"])
    g = df.groupby("video_id").agg(
        n_frames=("label", "size"),
        n_pos=("label", "sum"),
        orig_split=("orig_split", "first"),
    )
    g["pos_rate"] = g.n_pos / g.n_frames
    trans = (df.groupby("video_id").label
               .apply(lambda s: int((np.diff(s.values) != 0).sum()))
               .rename("n_transitions"))
    g = g.join(trans)
    g["regime"] = np.where(g.n_transitions == 0, "trimmed", "sweep")
    return g.reset_index()


def make_strata(pos_rate: pd.Series, min_per_stratum: int) -> pd.Series:
    """Bucket videos by standard-plane fraction for stratified regrouping.

    The buckets are derived from the data, not hard-coded. An earlier version
    used fixed cuts at 0.10 and 0.30 chosen for an assumed ~17% positive rate;
    the real rate is ~0.44, so every video landed in the top bucket and
    stratification silently stopped doing anything (audit S3). Fixed cuts
    encode an assumption about the class prior that this dataset does not meet.

    All-negative and all-positive videos get their own buckets because they are
    qualitatively different -- a video with no standard plane at all cannot
    contribute a positive to any fold -- and the interior is cut at quantiles.
    Buckets too small for StratifiedGroupKFold are merged into their neighbour,
    since a singleton stratum makes the split unsatisfiable.
    """
    s = pd.Series(index=pos_rate.index, dtype=object)
    s[pos_rate <= 0] = "none"
    s[pos_rate >= 1] = "all"
    interior = pos_rate[(pos_rate > 0) & (pos_rate < 1)]
    if len(interior):
        # up to 4 quantile bins, fewer if the values do not support them
        for n_bins in (4, 3, 2, 1):
            try:
                binned = pd.qcut(interior, n_bins, labels=False,
                                 duplicates="drop")
            except ValueError:
                continue
            if binned.notna().all():
                s[interior.index] = ["mixed%d" % int(b) for b in binned]
                break
        else:
            s[interior.index] = "mixed0"

    counts = s.value_counts()
    small = [k for k, v in counts.items() if v < min_per_stratum]
    if small:
        order = ["none"] + [f"mixed{i}" for i in range(4)] + ["all"]
        present = [k for k in order if k in set(counts.index)]
        for k in small:
            if k not in present or len(present) < 2:
                continue
            i = present.index(k)
            nb = present[i + 1] if i + 1 < len(present) else present[i - 1]
            log.info("  merging stratum %r (%d videos) into %r",
                     k, int(counts[k]), nb)
            s[s == k] = nb
    return s


def _report(df: pd.DataFrame, vids: pd.DataFrame, splits: dict) -> dict:
    """Print, and return, what each split actually contains."""
    diag = {}
    log.info("%-6s %6s %8s %9s %9s %8s %8s",
             "split", "videos", "frames", "pos-rate", "trimmed", "sweep",
             "trans/vid")
    for name in OFFICIAL_ORDER:
        vl = splits[name]
        sub = df[df.video_id.isin(vl)]
        vsub = vids[vids.video_id.isin(vl)]
        n_trim = int((vsub.regime == "trimmed").sum())
        n_swp = int((vsub.regime == "sweep").sum())
        tpv = float(vsub.n_transitions.mean()) if len(vsub) else 0.0
        log.info("%-6s %6d %8d %9.4f %9d %8d %8.2f",
                 name, len(vl), len(sub), sub.label.mean(), n_trim, n_swp, tpv)
        diag[name] = {
            "videos": len(vl), "frames": int(len(sub)),
            "pos_rate": round(float(sub.label.mean()), 4),
            "trimmed_videos": n_trim, "sweep_videos": n_swp,
            "transitions_per_video": round(tpv, 3),
        }
    return diag


def make_splits(index_csv: str, out_json: str, seed: int = 0,
                scheme: str = "official", test_frac: float = 0.2,
                val_frac: float = 0.2) -> dict:
    if scheme not in SCHEMES:
        raise SystemExit(f"unknown scheme {scheme!r}; pick from {SCHEMES}")

    df = pd.read_csv(index_csv)
    df = df[df.label >= 0].copy()
    if df.empty:
        raise SystemExit(f"{index_csv} has no labelled frames.")

    vids = video_table(df)
    log.info("%d labelled videos, %d frames, pooled pos-rate %.4f",
             len(vids), len(df), df.label.mean())
    log.info("regime by original folder:")
    for s in sorted(vids.orig_split.unique()):
        sub = vids[vids.orig_split == s]
        log.info("    %-6s %3d videos  %3d trimmed (0 transitions)  "
                 "%3d sweep  mean %.2f transitions/video",
                 s, len(sub), int((sub.regime == "trimmed").sum()),
                 int((sub.regime == "sweep").sum()), sub.n_transitions.mean())

    if scheme == "official":
        splits = _official_splits(vids)
    else:
        splits = _regrouped_splits(vids, seed, test_frac, val_frac)

    # ---- assert no leakage ----------------------------------------------
    s_tr, s_va, s_te = (set(splits[k]) for k in OFFICIAL_ORDER)
    assert not (s_tr & s_va), "train/val video overlap"
    assert not (s_tr & s_te), "train/test video overlap"
    assert not (s_va & s_te), "val/test video overlap"
    assert len(s_tr | s_va | s_te) == len(vids), "videos lost during splitting"
    for name in OFFICIAL_ORDER:
        if not splits[name]:
            raise SystemExit(f"the {name!r} split is empty")

    diag = _report(df, vids, splits)

    # ---- the temporal arm needs transitions in TRAIN --------------------
    train_trans = vids[vids.video_id.isin(s_tr)].n_transitions
    if int((train_trans > 0).sum()) == 0:
        log.warning("")
        log.warning("NO training video contains a label transition.")
        log.warning("A temporal model cannot learn when the label changes from "
                    "data where it never changes: a BiLSTM reaches zero "
                    "training loss by ignoring time and emitting one constant "
                    "per clip. The smoothed-CNN baseline, meanwhile, has its "
                    "transition prior tuned on validation, which does have "
                    "them.")
        log.warning("Arm 2 run on this split does not measure temporal "
                    "modelling. Enable splicing before running it; Arm 1 "
                    "is unaffected.")
        log.warning("")

    out = {
        **{k: sorted(splits[k]) for k in OFFICIAL_ORDER},
        "scheme": scheme,
        "seed": seed,
        "index_csv": str(index_csv),
        "n_videos": int(len(vids)),
        "diagnostics": diag,
        "train_has_transitions": bool((train_trans > 0).any()),
    }
    save_json(out, out_json)
    log.info("wrote %s (scheme=%s)", out_json, scheme)
    return out


def _official_splits(vids: pd.DataFrame) -> dict:
    """Use the dataset's own folders. All three carry labels in DatasetV3."""
    present = set(vids.orig_split.unique())
    missing = [s for s in OFFICIAL_ORDER if s not in present]
    if missing:
        raise SystemExit(
            f"scheme='official' needs labelled videos in every official folder, "
            f"but {missing} are absent (found {sorted(present)}). If this index "
            f"genuinely covers only one folder -- the synthetic smoke-test "
            f"fixture does -- pass --scheme regrouped.")
    extra = present - set(OFFICIAL_ORDER)
    if extra:
        raise SystemExit(f"unexpected orig_split value(s) {sorted(extra)}")

    log.info("scheme=official: using the dataset's own train/val/test folders")
    return {k: vids[vids.orig_split == k].video_id.tolist()
            for k in OFFICIAL_ORDER}


def _regrouped_splits(vids: pd.DataFrame, seed: int, test_frac: float,
                      val_frac: float) -> dict:
    """Pool every labelled video and re-partition, grouped by video.

    Kept for sensitivity analysis, not for headline numbers. On this dataset it
    mixes trimmed clips with untrimmed sweeps inside one test set -- see the
    module docstring.
    """
    n_test_folds = max(2, int(round(1 / test_frac)))
    n_val_folds = max(2, int(round(1 / val_frac)))

    regimes = set(vids.regime)
    if len(regimes) > 1:
        log.warning("scheme=regrouped pools %d trimmed clips with %d sweeps. "
                    "The resulting test set spans two populations and its "
                    "score is a seed-dependent weighted average of them. Use "
                    "--scheme official for headline numbers.",
                    int((vids.regime == "trimmed").sum()),
                    int((vids.regime == "sweep").sum()))

    v = vids.copy()
    v["stratum"] = make_strata(v.pos_rate, min_per_stratum=n_test_folds)
    log.info("strata: %s", v.stratum.value_counts().sort_index().to_dict())

    sgkf = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True,
                                random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(v, v.stratum, groups=v.video_id))

    tv = v.iloc[trainval_idx].reset_index(drop=True)
    test_vids = v.iloc[test_idx].video_id.tolist()

    sgkf2 = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True,
                                 random_state=seed + 1)
    tr_idx, va_idx = next(sgkf2.split(tv, tv.stratum, groups=tv.video_id))
    return {"train": tv.iloc[tr_idx].video_id.tolist(),
            "val": tv.iloc[va_idx].video_id.tolist(),
            "test": test_vids}


def load_splits(path: str | Path) -> dict:
    import json
    with open(path) as f:
        return json.load(f)


def frames_for(df: pd.DataFrame, splits: dict, which: str) -> pd.DataFrame:
    return df[df.video_id.isin(splits[which])].copy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out", default="data/splits.json")
    ap.add_argument("--scheme", default="official", choices=SCHEMES,
                    help="official: the dataset's own folders (default). "
                         "regrouped: pool and re-partition by video -- "
                         "sensitivity analysis only, see the module docstring.")
    ap.add_argument("--seed", type=int, default=0,
                    help="only affects --scheme regrouped")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()
    make_splits(a.index, a.out, a.seed, a.scheme, a.test_frac, a.val_frac)
