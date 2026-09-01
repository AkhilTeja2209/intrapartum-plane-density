"""Aggregate every run into the tables and figures the paper needs.

Two statistical points that decide whether the study says anything:

1. Compare models on the SAME test videos, paired. Two runs are evaluated on
   an identical test set, so the difference in metric can be bootstrapped by
   resampling videos and recomputing both models on the same resample. A
   paired test is far more powerful than comparing two independent confidence
   intervals, and "the CIs overlap" is not a valid way to conclude no
   difference.

2. Resample VIDEOS, not frames, and average over seeds. Frames within a video
   are dependent; seeds vary by more than people expect on datasets this size.
   A single-seed difference of 1-2 points on macro-F1 is usually noise.

    python -m src.analyze --results-dir results --out-dir report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import frame_metrics
from .utils import get_logger, load_json, save_json

log = get_logger("analyze")


def collect(results_dir: str | Path) -> pd.DataFrame:
    rows = []
    for rj in sorted(Path(results_dir).rglob("results.json")):
        d = load_json(rj)
        base = {
            "run": d["run"],
            "condition": d["condition"]["name"],
            "model_type": d["model_type"],
            "seed": d["seed"],
            "train_frames": d["train_stats"]["n_frames"],
            "train_videos": d["train_stats"]["n_videos"],
            "train_pos_rate": round(d["train_stats"]["pos_rate"], 4),
            "pred_csv": str(rj.parent / "test_predictions.csv"),
        }
        for variant, res in d["results"].items():
            rows.append({**base, "variant": variant,
                         **{k: v for k, v in res.items()
                            if isinstance(v, (int, float))}})
    if not rows:
        raise SystemExit(f"no results.json under {results_dir}")
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, metrics=("macro_f1", "balanced_accuracy",
                                         "auprc", "f1_standard",
                                         "top1_frame_precision")) -> pd.DataFrame:
    """Mean +/- std across seeds. This is the main results table."""
    g = df.groupby(["condition", "model_type", "variant", "train_frames",
                    "train_videos"])
    out = g[list(metrics)].agg(["mean", "std", "count"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.reset_index().sort_values(["model_type", "train_frames"])


def paired_bootstrap(pred_a: str, pred_b: str, thr_a: float = 0.5,
                     thr_b: float = 0.5, metric: str = "macro_f1",
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the difference (B - A) by resampling test videos.

    Both prediction files must cover the same test frames, which they do by
    construction: sampling is applied to train only.
    """
    a = pd.read_csv(pred_a).sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    b = pd.read_csv(pred_b).sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    if not a.video_id.equals(b.video_id) or not a.frame_idx.equals(b.frame_idx):
        raise ValueError("prediction files are not frame-aligned -- were they "
                         "evaluated on the same split?")

    m = a[["video_id", "frame_idx", "label"]].copy()
    m["pa"], m["pb"] = a.prob.values, b.prob.values

    obs = (frame_metrics(m.label.values, m.pb.values, thr_b)[metric]
           - frame_metrics(m.label.values, m.pa.values, thr_a)[metric])

    rng = np.random.default_rng(seed)
    vids = m.video_id.unique()
    groups = {v: g for v, g in m.groupby("video_id")}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(vids, size=len(vids), replace=True)
        s = pd.concat([groups[v] for v in pick], ignore_index=True)
        if s.label.nunique() < 2:
            continue
        diffs.append(frame_metrics(s.label.values, s.pb.values, thr_b)[metric]
                     - frame_metrics(s.label.values, s.pa.values, thr_a)[metric])
    diffs = np.asarray(diffs)

    # two-sided bootstrap p-value for H0: difference = 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "metric": metric,
        "observed_diff": float(obs),
        "ci95": [float(np.percentile(diffs, 2.5)),
                 float(np.percentile(diffs, 97.5))],
        "p_value": float(min(1.0, p)),
        "n_boot": int(len(diffs)),
    }


def density_curve(summary: pd.DataFrame, out_png: str) -> None:
    """Metric vs training frames. The headline figure of Protocol A."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (mt, var), g in summary.groupby(["model_type", "variant"]):
        g = g.sort_values("train_frames")
        ax.errorbar(g.train_frames, g.macro_f1_mean,
                    yerr=g.macro_f1_std.fillna(0), marker="o", capsize=3,
                    label=f"{mt} / {var}")
    ax.set_xscale("log")
    ax.set_xlabel("training frames (log scale)")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Standard-plane classification vs training frame density")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    log.info("wrote %s", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="report")
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                    help="Two run directory names to compare with a paired test")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = collect(args.results_dir)
    df.to_csv(out / "all_runs.csv", index=False)

    s = summarise(df)
    s.to_csv(out / "summary.csv", index=False)
    print(s.to_string(index=False))

    try:
        density_curve(s, str(out / "density_curve.png"))
    except ImportError:
        log.warning("matplotlib not installed; skipping the figure")

    if args.compare:
        a = Path(args.results_dir) / args.compare[0]
        b = Path(args.results_dir) / args.compare[1]
        ta = load_json(a / "results.json")["results"]["raw"]["threshold"]
        tb = load_json(b / "results.json")["results"]["raw"]["threshold"]
        r = paired_bootstrap(str(a / "test_predictions.csv"),
                             str(b / "test_predictions.csv"), ta, tb)
        print("\npaired video bootstrap  B - A")
        print(f"  A = {args.compare[0]}\n  B = {args.compare[1]}")
        print(f"  diff {r['observed_diff']:+.4f}  95% CI "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  p = {r['p_value']:.4f}")
        save_json(r, out / f"paired_{args.compare[0]}_vs_{args.compare[1]}.json")


if __name__ == "__main__":
    main()
