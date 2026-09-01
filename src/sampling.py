"""Frame sampling -- the independent variable of the study.

The research question is: *does the density with which frames are drawn from
video change what a classifier learns, holding the architecture fixed?*

To ask that cleanly we need a dial that turns "video dataset" into
"image dataset" without changing anything else -- same anatomy, same probe,
same scanners, same annotators, same label definition. Sampling k frames per
video from IUGC gives exactly that dial. k=1 is a synthetic image dataset:
one deliberately-chosen frame per case, which is how FETAL_PLANES_DB and every
other curated fetal-image corpus was actually built. k=all is the video
dataset. Everything in between traces the curve.

TWO PROTOCOLS, and the paper needs both:

  A. NATURAL  -- sparse = k frames from every video; dense = all frames from
     every video. This is the honest "image dataset vs video dataset"
     comparison, and it is what a practitioner cares about. It confounds
     total dataset size with density, deliberately, because in the real world
     those two come together.

  B. MATCHED-BUDGET -- both arms get the SAME number of training frames N.
     Sparse spends N on many videos at low density; dense spends N on fewer
     videos at high density. This decomposes A: if dense wins in A but loses
     in B, the gain was sample count, not density. If dense wins in both,
     density carries information that repeated sampling of the same case does
     not.

Protocol B is the contribution. Prior work runs A and reports it as if it
were B.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger("sampling")


# ---------------------------------------------------------------------------
# per-video selectors
# ---------------------------------------------------------------------------

def _uniform_positions(n: int, k: int) -> np.ndarray:
    """k evenly spaced indices spanning [0, n-1], endpoints included."""
    if k >= n:
        return np.arange(n)
    if k == 1:
        return np.array([n // 2])
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))


def sample_uniform(g: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    """Evenly spaced in time. Label-blind, so it does not smuggle in test
    information -- this is the honest sparse baseline."""
    g = g.sort_values("frame_idx")
    return g.iloc[_uniform_positions(len(g), k)]


def sample_random(g: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    k = min(k, len(g))
    idx = rng.choice(len(g), size=k, replace=False)
    return g.iloc[np.sort(idx)]


def sample_curated(g: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    """Mimic how a real image dataset is built: a sonographer freezes the frame
    when the plane looks right, plus some off-plane frames get captured too.

    Concretely: take the temporal centre of each contiguous run of standard-plane
    frames (that is the "best" frame of that acquisition), then fill the
    remaining budget with evenly spaced non-standard frames.

    This uses training labels only. It is applied to the TRAIN split, never to
    val or test.
    """
    g = g.sort_values("frame_idx").reset_index(drop=True)
    lab = g.label.values
    picks: list[int] = []

    # centre of each positive run
    i = 0
    while i < len(lab):
        if lab[i] == 1:
            j = i
            while j + 1 < len(lab) and lab[j + 1] == 1:
                j += 1
            picks.append((i + j) // 2)
            i = j + 1
        else:
            i += 1

    picks = picks[:k]
    remaining = k - len(picks)
    if remaining > 0:
        neg_pos = np.where(lab == 0)[0]
        if len(neg_pos):
            take = _uniform_positions(len(neg_pos), remaining)
            picks.extend(neg_pos[take].tolist())
    picks = sorted(set(picks))[:k]
    if not picks:  # degenerate: nothing selected, fall back to the middle frame
        picks = [len(g) // 2]
    return g.iloc[picks]


SELECTORS = {"uniform": sample_uniform, "random": sample_random,
             "curated": sample_curated}


# ---------------------------------------------------------------------------
# dataset-level protocols
# ---------------------------------------------------------------------------

def sample_per_video(df: pd.DataFrame, k: int | None, strategy: str = "uniform",
                     seed: int = 0) -> pd.DataFrame:
    """Take k frames from each video. k=None means keep everything."""
    if k is None:
        return df.copy()
    if strategy not in SELECTORS:
        raise ValueError(f"unknown strategy {strategy!r}; pick one of {list(SELECTORS)}")
    fn = SELECTORS[strategy]
    rng = np.random.default_rng(seed)
    out = [fn(g, k, rng) for _, g in df.groupby("video_id", sort=True)]
    return pd.concat(out, ignore_index=True)


def stride_sample(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Every stride-th frame of every video. A density dial that keeps
    temporal contiguity, unlike per-video-k."""
    if stride <= 1:
        return df.copy()
    out = []
    for _, g in df.groupby("video_id", sort=True):
        g = g.sort_values("frame_idx")
        out.append(g.iloc[::stride])
    return pd.concat(out, ignore_index=True)


def match_budget(df: pd.DataFrame, budget: int, seed: int = 0,
                 unit: str = "video") -> pd.DataFrame:
    """Trim to `budget` frames.

    unit='video' drops whole videos (keeps within-video density intact --
    this is the dense arm of Protocol B).
    unit='frame' drops random frames (keeps video count intact).
    """
    if len(df) <= budget:
        log.warning("budget %d >= available %d; returning everything", budget, len(df))
        return df.copy()

    rng = np.random.default_rng(seed)
    if unit == "frame":
        idx = rng.choice(len(df), size=budget, replace=False)
        return df.iloc[np.sort(idx)].reset_index(drop=True)

    vids = df.video_id.unique().tolist()
    rng.shuffle(vids)

    kept, total, leftover = [], 0, None
    for v in vids:
        n = int((df.video_id == v).sum())
        if total + n <= budget:
            kept.append(v)
            total += n
        else:
            leftover = v
            break

    parts = [df[df.video_id.isin(kept)]] if kept else []

    # Top up to the budget EXACTLY with a contiguous chunk of the next video.
    # Contiguous rather than random, so within-video temporal density -- the
    # whole point of the dense arm -- survives the trim. Protocol B is only
    # a fair test if the two arms get the same frame count, so overshooting
    # or undershooting here quietly weakens the comparison.
    need = budget - total
    if need > 0 and leftover is not None:
        g = df[df.video_id == leftover].sort_values("frame_idx")
        start = int(rng.integers(0, max(1, len(g) - need + 1)))
        parts.append(g.iloc[start:start + need])

    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[:budget].copy()
    return out.reset_index(drop=True)


def build_condition(train_df: pd.DataFrame, cond: dict, seed: int = 0) -> pd.DataFrame:
    """Turn a condition spec from the config into an actual training frame set.

    Spec keys:
        k        : frames per video, or None for all
        strategy : uniform | random | curated  (only used when k is set)
        stride   : keep every stride-th frame (applied after k)
        budget   : cap total frames
        budget_unit : 'video' or 'frame'
    """
    out = sample_per_video(train_df, cond.get("k"), cond.get("strategy", "uniform"), seed)
    if cond.get("stride", 1) > 1:
        out = stride_sample(out, cond["stride"])
    if cond.get("budget"):
        out = match_budget(out, cond["budget"], seed, cond.get("budget_unit", "video"))
    log.info("condition %-22s -> %6d frames | %3d videos | pos-rate %.4f",
             cond.get("name", "?"), len(out), out.video_id.nunique(),
             out.label.mean())
    return out.reset_index(drop=True)


def describe(df: pd.DataFrame) -> dict:
    return {
        "n_frames": int(len(df)),
        "n_videos": int(df.video_id.nunique()),
        "pos_rate": float(df.label.mean()),
        "frames_per_video_median": float(df.groupby("video_id").size().median()),
    }
