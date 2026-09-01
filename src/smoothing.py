"""Post-hoc temporal smoothing of frame-wise predictions.

This module exists to make the study honest.

A CNN-LSTM beating a raw frame-wise ResNet does NOT show that the model
learned temporal reasoning. Standard planes occur in contiguous runs, so
almost all of the exploitable temporal structure is just label
autocorrelation, and a three-line moving average over the frame-wise
probabilities captures it -- with zero extra parameters, zero extra training,
and no video pipeline.

So the real baseline for the temporal arm is:

    frame-wise ResNet-18  +  smoothing over its own output probabilities

If the LSTM does not beat THAT, the paper's honest conclusion is that temporal
modelling bought nothing that a post-processing filter does not already
provide. That is a publishable negative result and a much stronger one than
a comparison against an unsmoothed baseline that nobody would deploy.

The Viterbi option is the strongest of the three: it is an explicit two-state
Markov chain over {non-standard, standard} whose transition matrix encodes
"runs are long", which is exactly the prior an LSTM would have to learn from
data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average(p: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1:
        return p.copy()
    k = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(p, pad, mode="edge"), k, mode="valid")[:len(p)]


def median_filter(p: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1:
        return p.copy()
    pad = w // 2
    q = np.pad(p, pad, mode="edge")
    return np.array([np.median(q[i:i + w]) for i in range(len(p))])


def viterbi_smooth(p: np.ndarray, p_stay: float = 0.95,
                   eps: float = 1e-6) -> np.ndarray:
    """Two-state HMM MAP decode over one video's probability sequence.

    Emission  P(obs | state=1) = p, P(obs | state=0) = 1-p
    Transition  stay = p_stay, switch = 1 - p_stay

    Returns a hard 0/1 sequence as floats, so it slots into the same metric
    code as a probability (AUROC/AUPRC on a binary vector is degenerate, so
    compare this arm on threshold-based metrics only).
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    n = len(p)
    if n == 0:
        return p

    log_e = np.stack([np.log(1 - p), np.log(p)], axis=1)      # (n,2)
    log_t = np.log(np.array([[p_stay, 1 - p_stay],
                             [1 - p_stay, p_stay]]))
    delta = np.zeros((n, 2))
    psi = np.zeros((n, 2), dtype=int)
    delta[0] = np.log(np.array([0.5, 0.5])) + log_e[0]

    for t in range(1, n):
        for s in range(2):
            scores = delta[t - 1] + log_t[:, s]
            psi[t, s] = int(np.argmax(scores))
            delta[t, s] = scores[psi[t, s]] + log_e[t, s]

    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(n - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path.astype(float)


SMOOTHERS = {
    "none": lambda p, **kw: p,
    "movavg": lambda p, w=5, **kw: moving_average(p, w),
    "median": lambda p, w=5, **kw: median_filter(p, w),
    "viterbi": lambda p, p_stay=0.95, **kw: viterbi_smooth(p, p_stay),
}


def smooth_by_video(df: pd.DataFrame, prob: np.ndarray, method: str = "movavg",
                    **kw) -> np.ndarray:
    """Apply a smoother independently within each video, in frame order.

    Smoothing across a video boundary would blend two unrelated acquisitions,
    so the grouping is not optional.
    """
    if method not in SMOOTHERS:
        raise ValueError(f"unknown smoother {method!r}; pick from {list(SMOOTHERS)}")
    fn = SMOOTHERS[method]

    d = df.reset_index(drop=True).copy()
    d["_prob"] = np.asarray(prob, dtype=float)
    d["_row"] = np.arange(len(d))
    out = np.empty(len(d), dtype=float)

    for _, g in d.groupby("video_id", sort=False):
        g = g.sort_values("frame_idx")
        out[g._row.values] = fn(g._prob.values, **kw)
    return out


def tune_smoother(val_df: pd.DataFrame, val_prob: np.ndarray,
                  objective: str = "macro_f1") -> dict:
    """Grid-search the smoother AND its threshold on validation.

    The smoothed baseline gets the same tuning budget as the LSTM -- otherwise
    we would be handicapping it and the comparison would be rigged in the
    direction we are hoping for, which is the failure mode this whole module
    is meant to prevent.
    """
    from .metrics import pick_threshold, frame_metrics

    best = {"score": -1.0}
    grid = ([{"method": "none"}]
            + [{"method": "movavg", "w": w} for w in (3, 5, 9, 15, 25)]
            + [{"method": "median", "w": w} for w in (3, 5, 9, 15, 25)]
            + [{"method": "viterbi", "p_stay": s} for s in (0.90, 0.95, 0.98, 0.99)])

    for cfg in grid:
        kw = {k: v for k, v in cfg.items() if k != "method"}
        sm = smooth_by_video(val_df, val_prob, cfg["method"], **kw)
        thr = pick_threshold(val_df.label.values, sm, objective)
        score = frame_metrics(val_df.label.values, sm, thr)[objective]
        if score > best["score"]:
            best = {"score": float(score), "threshold": thr, **cfg}
    return best
