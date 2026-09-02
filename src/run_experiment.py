"""Run one condition x one seed, end to end, and write a results row.

Every experiment in the study goes through this one entry point so that no
condition can accidentally get a different recipe than another.

Examples
--------
# Arm 1, sparse (synthetic "image dataset"): 1 frame per video
python -m src.run_experiment --condition sparse_k1 --seed 0

# Arm 1, dense (video-derived): every frame
python -m src.run_experiment --condition dense_all --seed 0

# Arm 1, Protocol B: dense but capped to the sparse arm's frame budget
python -m src.run_experiment --condition dense_matched_k5 --seed 0

# Arm 2, temporal
python -m src.run_experiment --condition dense_all --model temporal --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from .datasets import (ClipDataset, FrameDataset, SplicedClipDataset,
                       class_weights)
from .engine import fit, make_loader, predict
from .metrics import bootstrap_ci, evaluate_all, pick_threshold
from .models import build, load_encoder_from
from .sampling import build_condition, describe
from .smoothing import smooth_by_video, tune_smoother
from .splits import load_splits
from .utils import count_params, get_logger, save_json, set_seed

log = get_logger("run")


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--model", choices=["frame", "temporal"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm-start", default=None,
                    help="Checkpoint whose encoder initialises the temporal arm")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.model:
        cfg.setdefault("model", {})["type"] = args.model
    temporal = cfg.get("model", {}).get("type", "frame") == "temporal"

    conds = {c["name"]: c for c in cfg["conditions"]}
    if args.condition not in conds:
        raise SystemExit(f"unknown condition {args.condition!r}. "
                         f"Available: {sorted(conds)}")
    cond = conds[args.condition]

    run_name = f"{args.condition}__{'temporal' if temporal else 'frame'}" \
               f"{'_' + args.tag if args.tag else ''}__seed{args.seed}"
    out_dir = Path(args.out_dir or cfg["paths"]["results_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_ = get_logger("run", str(out_dir / "log.txt"))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_.info("run=%s device=%s", run_name, device)

    # ---- data ------------------------------------------------------------
    index = pd.read_csv(cfg["paths"]["index_csv"])
    index = index[index.label >= 0]
    splits = load_splits(cfg["paths"]["splits_json"])
    frames_dir = cfg["paths"]["frames_dir"]

    train_full = index[index.video_id.isin(splits["train"])]
    val_df = index[index.video_id.isin(splits["val"])] \
        .sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    test_df = index[index.video_id.isin(splits["test"])] \
        .sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    # Sampling is applied to TRAIN ONLY. Validation and test are always the
    # complete, dense frame sets -- every condition is scored on exactly the
    # same frames, or the numbers are not comparable.
    train_df = build_condition(train_full, cond, seed=args.seed)
    train_df = train_df.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)

    log_.info("train %s", describe(train_df))
    log_.info("val   %s", describe(val_df))
    log_.info("test  %s", describe(test_df))

    if temporal and cond.get("k") is not None:
        log_.warning("temporal arm on a sparsely-sampled condition: clips will "
                     "span large real-time gaps, so 'temporal context' is not "
                     "what it sounds like. This is only meaningful as an "
                     "ablation.")

    # ---- datasets --------------------------------------------------------
    img = cfg["data"]["img_size"]
    bs = cfg["train"]["batch_size"]
    nw = cfg["data"].get("workers", 4)

    if temporal:
        T = cfg["data"]["clip_len"]
        # Splicing synthesises the label transitions the official train split
        # does not contain. splice_p=0 recovers the
        # unspliced baseline, which is the ablation the paper needs.
        splice_p = float(cfg["data"].get("splice_p", 0.0))
        if splice_p > 0:
            tr_ds = SplicedClipDataset(
                train_df, frames_dir, T, cfg["data"]["clip_stride"], img,
                cfg["data"].get("temporal_stride", 1), splice_p, args.seed,
                cfg["data"].get("splice_same_session_p", 0.8))
            log_.info("splicing ON: p=%.2f, %d positive / %d negative train "
                      "videos available as partners", splice_p,
                      len(tr_ds.pos_vids), len(tr_ds.neg_vids))
        else:
            tr_ds = ClipDataset(train_df, frames_dir, T,
                                cfg["data"]["clip_stride"], img, True,
                                cfg["data"].get("temporal_stride", 1))
            log_.warning("splicing OFF and the training split has no label "
                         "transitions -- the temporal arm cannot learn "
                         "transitions from this data. Baseline/ablation only.")
        va_ds = ClipDataset(val_df, frames_dir, T, T // 2, img, False,
                            cfg["data"].get("temporal_stride", 1))
        te_ds = ClipDataset(test_df, frames_dir, T, T // 2, img, False,
                            cfg["data"].get("temporal_stride", 1))
        # keep GPU memory comparable: a clip is T images
        bs = max(1, bs // T)
    else:
        tr_ds = FrameDataset(train_df, frames_dir, img, True)
        va_ds = FrameDataset(val_df, frames_dir, img, False)
        te_ds = FrameDataset(test_df, frames_dir, img, False)

    tr_ld = make_loader(tr_ds, bs, True, nw)
    va_ld = make_loader(va_ds, bs, False, nw)
    te_ld = make_loader(te_ds, bs, False, nw)

    # ---- model -----------------------------------------------------------
    model = build(cfg).to(device)
    if temporal and args.warm_start:
        miss, unexp = load_encoder_from(model, args.warm_start)
        log_.info("warm-started encoder from %s (missing=%d unexpected=%d)",
                  args.warm_start, len(miss), len(unexp))
    log_.info("trainable params: %s", f"{count_params(model):,}")

    cw = class_weights(train_df) if cfg["train"].get("class_weighted", True) else None

    # ---- train -----------------------------------------------------------
    model, best_val, history = fit(model, tr_ld, va_ld, val_df, cfg["train"],
                                   device, temporal, cw)
    torch.save({"model": model.state_dict(), "cfg": cfg, "cond": cond,
                "seed": args.seed}, out_dir / "best.pt")

    # ---- evaluate --------------------------------------------------------
    val_prob = predict(model, va_ld, device, temporal, len(val_df))
    test_prob = predict(model, te_ld, device, temporal, len(test_df))

    thr = pick_threshold(val_df.label.values, val_prob, "macro_f1")
    results = {"raw": evaluate_all(test_df, test_prob, thr)}
    lo, hi = bootstrap_ci(test_df, test_prob, thr, "macro_f1",
                          cfg["eval"].get("n_boot", 1000), args.seed)
    results["raw"]["macro_f1_ci95"] = [lo, hi]

    # The smoothed frame-wise result is the number the temporal arm must beat.
    if not temporal and cfg["eval"].get("smoothing", True):
        best_sm = tune_smoother(val_df, val_prob, "macro_f1")
        kw = {k: v for k, v in best_sm.items()
              if k not in ("method", "score", "threshold")}
        sm_test = smooth_by_video(test_df, test_prob, best_sm["method"], **kw)
        results["smoothed"] = evaluate_all(test_df, sm_test, best_sm["threshold"])
        results["smoothed"]["smoother"] = best_sm
        log_.info("smoothed (%s) test macroF1 %.4f vs raw %.4f",
                  best_sm["method"], results["smoothed"]["macro_f1"],
                  results["raw"]["macro_f1"])

    payload = {
        "run": run_name, "condition": cond, "seed": args.seed,
        "model_type": "temporal" if temporal else "frame",
        "model_cfg": cfg.get("model", {}),
        "train_stats": describe(train_df),
        "best_val": best_val, "history": history, "results": results,
    }
    save_json(payload, out_dir / "results.json")
    pd.DataFrame({"video_id": test_df.video_id, "frame_idx": test_df.frame_idx,
                  "label": test_df.label, "prob": test_prob}) \
        .to_csv(out_dir / "test_predictions.csv", index=False)

    log_.info("TEST macroF1 %.4f [%.4f, %.4f] balAcc %.4f auprc %.4f top1 %.4f",
              results["raw"]["macro_f1"], lo, hi,
              results["raw"]["balanced_accuracy"], results["raw"]["auprc"],
              results["raw"]["top1_frame_precision"])
    print(json.dumps(results["raw"], indent=2))


if __name__ == "__main__":
    main()
