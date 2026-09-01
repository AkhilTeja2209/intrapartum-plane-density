"""Metrics.

Plain accuracy is close to useless here: with ~15-20% standard-plane frames,
"always say non-standard" already scores in the low 80s. Every headline number
in the paper should be balanced accuracy, macro-F1, or AUPRC.

There are also two metrics that plain frame-level scoring misses entirely,
and both are closer to what the model is actually for:

  * top1_frame_precision -- take the single highest-scoring frame in each
    video and ask whether it is truly a standard plane. That is literally the
    clinical workflow: the tool proposes one frame for the clinician to
    measure AoP/HSD on. A model can have mediocre frame-level F1 and still be
    perfect at this, or vice versa.

  * video_detection_rate -- in what fraction of videos that contain a standard
    plane does the model flag at least one? Missing a case entirely is a
    different failure from mislabelling a few frames within a case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)


def frame_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    y_pred = (prob >= thr).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    out = {
        "threshold": float(thr),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_standard": float(f1),
        "precision_standard": float(p),
        "recall_standard": float(r),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "accuracy": float((y_pred == y_true).mean()),
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
    }
    if len(set(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, prob))
        out["auprc"] = float(average_precision_score(y_true, prob))
    else:
        out["auroc"] = out["auprc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    return out


def video_metrics(df: pd.DataFrame, prob: np.ndarray, thr: float = 0.5) -> dict:
    """df must carry video_id and label, aligned row-for-row with prob."""
    d = df[["video_id", "label"]].copy().reset_index(drop=True)
    d["prob"] = np.asarray(prob, dtype=float)
    d["pred"] = (d.prob >= thr).astype(int)

    hits, dets, considered = [], [], 0
    for _, g in d.groupby("video_id"):
        if g.label.sum() == 0:
            continue                       # no ground-truth plane to find
        considered += 1
        hits.append(int(g.loc[g.prob.idxmax(), "label"] == 1))
        dets.append(int(g.pred.sum() > 0))

    # false alarms on videos that contain no standard plane at all
    fa = [int(g.pred.sum() > 0)
          for _, g in d.groupby("video_id") if g.label.sum() == 0]

    return {
        "top1_frame_precision": float(np.mean(hits)) if hits else float("nan"),
        "video_detection_rate": float(np.mean(dets)) if dets else float("nan"),
        "video_false_alarm_rate": float(np.mean(fa)) if fa else float("nan"),
        "n_videos_with_plane": considered,
        "n_videos_without_plane": len(fa),
    }


def pick_threshold(y_true: np.ndarray, prob: np.ndarray,
                   objective: str = "macro_f1") -> float:
    """Choose the operating point on VALIDATION only, then freeze it.

    Sweeping the threshold on test and reporting the best is a real and common
    way to manufacture a difference between two models that are actually tied.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    grid = np.unique(np.round(np.linspace(0.01, 0.99, 99), 4))
    best, best_thr = -1.0, 0.5
    for t in grid:
        pred = (prob >= t).astype(int)
        if objective == "macro_f1":
            s = f1_score(y_true, pred, average="macro", zero_division=0)
        elif objective == "balanced_accuracy":
            s = balanced_accuracy_score(y_true, pred)
        elif objective == "f1_standard":
            s = f1_score(y_true, pred, zero_division=0)
        else:
            raise ValueError(objective)
        if s > best:
            best, best_thr = s, float(t)
    return best_thr


def evaluate_all(df: pd.DataFrame, prob: np.ndarray, thr: float) -> dict:
    out = frame_metrics(df.label.values, prob, thr)
    out.update(video_metrics(df, prob, thr))
    return out


def bootstrap_ci(df: pd.DataFrame, prob: np.ndarray, thr: float,
                 metric: str = "macro_f1", n_boot: int = 1000,
                 seed: int = 0) -> tuple[float, float]:
    """Resample VIDEOS, not frames.

    Frames within a video are strongly dependent, so a frame-level bootstrap
    reports intervals several times narrower than reality and will make two
    indistinguishable models look significantly different.
    """
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True).copy()
    d["prob"] = prob
    vids = d.video_id.unique()
    groups = {v: g for v, g in d.groupby("video_id")}

    vals = []
    for _ in range(n_boot):
        pick = rng.choice(vids, size=len(vids), replace=True)
        s = pd.concat([groups[v] for v in pick], ignore_index=True)
        if s.label.nunique() < 2:
            continue
        vals.append(frame_metrics(s.label.values, s.prob.values, thr)[metric])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
