"""Split by VIDEO, never by frame.

This is the single most important file in the repo. Consecutive ultrasound
frames are near-duplicates: frame 340 and frame 341 differ by ~30 ms of probe
motion. If a random frame-level split puts 340 in train and 341 in test, the
model is scored on images it has effectively memorised, and reported accuracy
inflates by roughly 10-20 points. Every number in the study would be wrong in
the same direction, so the inflation would not even show up as noise.

We therefore split on video_id with sklearn's GroupShuffleSplit, and we
stratify on a coarse video-level bucket (what fraction of that video's frames
are standard planes) so that all three splits see a similar mix of
easy/hard videos.

The split is written once to disk and reused by every condition and seed.
Conditions must differ ONLY in how training frames are sampled -- if each
condition re-drew its own split, the comparison would confound sampling
density with which videos happened to land in test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .utils import get_logger, save_json

log = get_logger("splits")


def video_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per video: frame count and standard-plane fraction."""
    g = df.groupby("video_id").agg(
        n_frames=("label", "size"),
        n_pos=("label", "sum"),
        orig_split=("orig_split", "first"),
    )
    g["pos_rate"] = g.n_pos / g.n_frames
    return g.reset_index()


def stratum(pos_rate: float) -> int:
    """Coarse buckets. Fine-grained strata would leave singleton groups that
    StratifiedGroupKFold cannot place."""
    if pos_rate == 0:
        return 0
    if pos_rate < 0.10:
        return 1
    if pos_rate < 0.30:
        return 2
    return 3


def make_splits(index_csv: str, out_json: str, seed: int = 0,
                test_frac: float = 0.2, val_frac: float = 0.2) -> dict:
    df = pd.read_csv(index_csv)
    df = df[df.label >= 0].copy()  # drop the unlabelled official test folder

    vids = video_table(df)
    vids["stratum"] = vids.pos_rate.map(stratum)
    log.info("%d labelled videos, stratum sizes: %s",
             len(vids), vids.stratum.value_counts().sort_index().to_dict())

    # StratifiedGroupKFold with n_splits = 1/test_frac gives us a test fold
    # that is both group-disjoint and stratum-balanced.
    n_test_folds = max(2, int(round(1 / test_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True,
                                random_state=seed)
    trainval_idx, test_idx = next(sgkf.split(vids, vids.stratum, groups=vids.video_id))

    tv = vids.iloc[trainval_idx].reset_index(drop=True)
    test_vids = vids.iloc[test_idx].video_id.tolist()

    n_val_folds = max(2, int(round(1 / val_frac)))
    sgkf2 = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True,
                                 random_state=seed + 1)
    tr_idx, va_idx = next(sgkf2.split(tv, tv.stratum, groups=tv.video_id))
    train_vids = tv.iloc[tr_idx].video_id.tolist()
    val_vids = tv.iloc[va_idx].video_id.tolist()

    splits = {"train": sorted(train_vids), "val": sorted(val_vids),
              "test": sorted(test_vids), "seed": seed}

    # ---- assert no leakage ----------------------------------------------
    s_tr, s_va, s_te = map(set, (train_vids, val_vids, test_vids))
    assert not (s_tr & s_va), "train/val video overlap"
    assert not (s_tr & s_te), "train/test video overlap"
    assert not (s_va & s_te), "val/test video overlap"
    assert len(s_tr | s_va | s_te) == len(vids), "videos lost during splitting"

    for name, vl in (("train", train_vids), ("val", val_vids), ("test", test_vids)):
        sub = df[df.video_id.isin(vl)]
        log.info("%-5s %3d videos %6d frames  pos-rate %.4f",
                 name, len(vl), len(sub), sub.label.mean())

    save_json(splits, out_json)
    log.info("wrote %s", out_json)
    return splits


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--val-frac", type=float, default=0.2)
    a = ap.parse_args()
    make_splits(a.index, a.out, a.seed, a.test_frac, a.val_frac)
