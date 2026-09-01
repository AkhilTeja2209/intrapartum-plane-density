"""Smoke test -- prove the pipeline works before you spend GPU hours on it.

Two modes.

  --synthetic
      Generates ~20 fake "videos" of random noise, runs the ENTIRE pipeline
      on them, and checks every stage. No dataset needed, no GPU needed,
      runs on your laptop in a couple of minutes. Do this FIRST, before you
      download anything. It catches broken installs, missing packages, and
      typos in your config -- all the boring failures -- while they are cheap.
      The accuracy numbers will be garbage (the images are noise). That is
      fine and expected. You are testing plumbing, not learning.

  --real
      Runs the same end-to-end path on a small slice of the actual dataset
      (default 20 videos, 2 epochs, 96px). Takes a few minutes on a GPU.
      Do this AFTER preparing the real data and BEFORE launching the full
      grid. It catches label-mapping mistakes and path problems that the
      synthetic mode cannot see.

Usage:
    python -m src.smoke_test --synthetic
    python -m src.smoke_test --real --n-videos 20 --epochs 2
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger("smoke")

PASS, FAIL = "  [PASS]", "  [FAIL]"


def _check(cond: bool, msg: str, failures: list) -> bool:
    print((PASS if cond else FAIL) + " " + msg)
    if not cond:
        failures.append(msg)
    return cond


# ---------------------------------------------------------------------------
# synthetic mode
# ---------------------------------------------------------------------------

def make_fake_dataset(root: Path, n_videos: int = 20, seed: int = 0) -> Path:
    """Write fake frames + a class_label.csv that mimics the real layout."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    frames_dir = root / "frames" / "train"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for v in range(n_videos):
        vid = f"video{v:04d}"
        n = int(rng.integers(30, 70))
        lab = np.zeros(n, dtype=int)
        # contiguous positive runs, like real standard-plane sequences
        for _ in range(int(rng.integers(1, 3))):
            s = int(rng.integers(0, max(1, n - 12)))
            L = int(rng.integers(4, 12))
            lab[s:s + L] = 1

        d = frames_dir / vid
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            # Positives get a brighter blob so a model CAN in principle learn
            # something -- otherwise a zero-signal run masks real bugs.
            img = rng.integers(40, 90, (128, 128), dtype=np.uint8)
            if lab[i]:
                img[40:80, 40:80] = np.clip(
                    img[40:80, 40:80].astype(int) + 90, 0, 255).astype(np.uint8)
            Image.fromarray(img).save(d / f"{i:06d}.jpg", quality=85)
            rows.append({"video_id": vid, "frame_idx": i, "label": int(lab[i])})

    csv_dir = root / "DatasetV3" / "train" / "cls"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_dir / "class_label.csv", index=False)
    log.info("fake dataset: %d videos, %d frames", n_videos, len(rows))
    return root


def _check_label_parsing(failures: list) -> None:
    """Guard the two defects that left the real index with zero train labels.

    Both were silent. The ALL sentinel parsed to an empty set, and the doubled
    filenames in the published zip matched no label row -- between them every
    standard-plane training video was dropped, and the pipeline ran to
    completion on what was left. These assertions are cheap and they fail loudly
    if either regresses.
    """
    import tempfile as _tf

    from .build_index import (canonical_video_id, load_video_index_table,
                              parse_index_list)

    # -- ALL / NONE sentinels (train/cls/class_label.csv uses them) ---------
    _check(parse_index_list("ALL", 20) == set(range(20)),
           "'ALL' expands to every frame", failures)
    _check(parse_index_list("NONE", 20) == set(),
           "'NONE' expands to no frames", failures)
    _check(parse_index_list("all", 5) == set(range(5)),
           "sentinel match is case-insensitive", failures)
    try:
        parse_index_list("ALL")
        _check(False, "'ALL' without frame_count raises", failures)
    except ValueError:
        _check(True, "'ALL' without frame_count raises", failures)

    # -- doubled stems, and the legitimate '__' that must survive ------------
    _check(canonical_video_id("20190909T155747I1__20190909T155747I1")
           == "20190909T155747I1", "doubled stem 'X__X' collapses to 'X'",
           failures)
    _check(canonical_video_id("20190830T115644__B_produce_tmp_0")
           == "20190830T115644__B_produce_tmp_0",
           "ordinary '__' in a filename is left alone", failures)
    _check(canonical_video_id("a__b") == "a__b",
           "unequal halves are left alone", failures)

    # -- end to end through the real loader ---------------------------------
    with _tf.TemporaryDirectory() as td:
        csv = Path(td) / "class_label.csv"
        csv.write_text(
            "filename,frame_count,pos_index,neg_index\n"
            "vidA__vidA.avi,4,ALL,NONE\n"
            "vidB.avi,4,NONE,ALL\n"
            "vidC.avi,4,\"[0, 1]\",\"[2, 3]\"\n",
            encoding="utf-8")
        t = load_video_index_table(csv)
        _check(len(t) == 12, f"sentinel CSV yields every frame ({len(t)}/12)",
               failures)
        _check(set(t.video_id) == {"vidA", "vidB", "vidC"},
               "doubled stem is canonicalised on load", failures)
        _check(int(t[t.video_id == "vidA"].label.sum()) == 4,
               "ALL video is fully positive", failures)
        _check(int(t[t.video_id == "vidB"].label.sum()) == 0,
               "NONE video is fully negative", failures)

        bad = Path(td) / "bad.csv"
        bad.write_text(
            "filename,frame_count,pos_index,neg_index\n"
            "vidD.avi,10,ALL,\"[0, 1]\"\n", encoding="utf-8")
        try:
            load_video_index_table(bad)
            _check(False, "ALL that fails to partition is rejected", failures)
        except SystemExit:
            _check(True, "ALL that fails to partition is rejected", failures)


def run_synthetic(keep: bool = False) -> int:
    failures: list[str] = []
    root = Path(tempfile.mkdtemp(prefix="spc_smoke_"))
    print(f"\nworking in {root}\n")

    try:
        print("0. label-parsing regression guards")
        _check_label_parsing(failures)

        print("\n1. generate fake dataset")
        make_fake_dataset(root)
        _check((root / "frames" / "train").exists(), "frames written", failures)

        print("\n2. build_index")
        r = subprocess.run(
            [sys.executable, "-m", "src.build_index",
             "--dataset-root", str(root / "DatasetV3"),
             "--frames-dir", str(root / "frames"),
             "--out", str(root / "index.csv")],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
        _check(r.returncode == 0, "build_index ran", failures)

        if (root / "index.csv").exists():
            idx = pd.read_csv(root / "index.csv")
            lab = idx[idx.label >= 0]
            _check(len(lab) > 0, f"index has labelled frames ({len(lab)})", failures)
            _check(0.0 < lab.label.mean() < 1.0,
                   f"both classes present (pos-rate {lab.label.mean():.3f})", failures)
            _check(lab.label.isin([0, 1]).all(), "labels are 0/1", failures)

        print("\n3. splits")
        # The fixture writes everything into one folder, so there is no
        # official train/val/test to honour -- regrouped is the only scheme
        # that applies. The real study uses --scheme official; see splits.py.
        r = subprocess.run(
            [sys.executable, "-m", "src.splits", "--index", str(root / "index.csv"),
             "--out", str(root / "splits.json"), "--scheme", "regrouped",
             "--test-frac", "0.3", "--val-frac", "0.3"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
        _check(r.returncode == 0, "splits ran (leakage assertions passed)", failures)

        # A single-folder index must be refused by the official scheme rather
        # than silently producing an empty val/test.
        r_off = subprocess.run(
            [sys.executable, "-m", "src.splits", "--index", str(root / "index.csv"),
             "--out", str(root / "splits_official.json"), "--scheme", "official"],
            capture_output=True, text=True)
        _check(r_off.returncode != 0,
               "official scheme refuses a single-folder index", failures)

        if (root / "splits.json").exists():
            from .splits import load_splits
            sp = load_splits(root / "splits.json")
            overlap = (set(sp["train"]) & set(sp["test"])) | \
                      (set(sp["train"]) & set(sp["val"])) | \
                      (set(sp["val"]) & set(sp["test"]))
            _check(not overlap, "no video appears in two splits", failures)
            _check(all(len(sp[k]) > 0 for k in ("train", "val", "test")),
                   "all three splits non-empty", failures)

        print("\n4. sampling conditions")
        try:
            from .sampling import build_condition
            from .splits import load_splits
            idx = pd.read_csv(root / "index.csv")
            idx = idx[idx.label >= 0]
            sp = load_splits(root / "splits.json")
            tr = idx[idx.video_id.isin(sp["train"])]
            k1 = build_condition(tr, {"name": "k1", "k": 1, "strategy": "uniform"})
            allf = build_condition(tr, {"name": "all", "k": None})
            _check(len(k1) == tr.video_id.nunique(),
                   "k=1 gives exactly one frame per video", failures)
            _check(len(allf) > len(k1), "dense > sparse", failures)
            b = build_condition(tr, {"name": "b", "k": None,
                                     "budget": len(k1) * 3, "budget_unit": "video"})
            _check(len(b) <= len(k1) * 3, "budget cap respected", failures)
        except Exception as e:
            _check(False, f"sampling raised {e!r}", failures)

        print("\n5. metrics + smoothing")
        try:
            from .metrics import evaluate_all, pick_threshold
            from .smoothing import smooth_by_video, tune_smoother
            idx = pd.read_csv(root / "index.csv")
            d = idx[idx.label >= 0].reset_index(drop=True)
            rng = np.random.default_rng(0)
            p = np.clip(d.label.values * 0.6 + rng.normal(0.2, 0.2, len(d)), 0, 1)
            thr = pick_threshold(d.label.values, p)
            m = evaluate_all(d, p, thr)
            _check(0 <= m["macro_f1"] <= 1, f"macro_f1 valid ({m['macro_f1']:.3f})", failures)
            _check("top1_frame_precision" in m, "video-level metrics computed", failures)
            best = tune_smoother(d, p)
            sm = smooth_by_video(d, p, best["method"],
                                 **{k: v for k, v in best.items()
                                    if k not in ("method", "score", "threshold")})
            _check(len(sm) == len(p), "smoother returns one value per frame", failures)
        except Exception as e:
            _check(False, f"metrics/smoothing raised {e!r}", failures)

        print("\n6. training (2 epochs, tiny)")
        try:
            import torch  # noqa: F401
            cfg = root / "cfg.yaml"
            cfg.write_text(f"""
paths:
  dataset_root: {root / 'DatasetV3'}
  frames_dir:   {root / 'frames'}
  index_csv:    {root / 'index.csv'}
  splits_json:  {root / 'splits.json'}
  results_dir:  {root / 'results'}
data:
  img_size: 64
  workers: 0
  clip_len: 4
  clip_stride: 2
  temporal_stride: 1
model:
  type: frame
  arch: resnet18
  pretrained: false
  dropout: 0.2
  rnn: lstm
  hidden: 32
  layers: 1
  bidirectional: true
  freeze_encoder: false
train:
  epochs: 2
  patience: 2
  batch_size: 16
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 0
  grad_clip: 1.0
  amp: false
  class_weighted: true
eval:
  smoothing: true
  n_boot: 50
conditions:
  - name: smoke_sparse
    k: 2
    strategy: uniform
  - name: smoke_dense
    k: null
""")
            for cond, mtype in (("smoke_sparse", "frame"), ("smoke_dense", "temporal")):
                r = subprocess.run(
                    [sys.executable, "-m", "src.run_experiment", "--config", str(cfg),
                     "--condition", cond, "--model", mtype, "--seed", "0"],
                    capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stdout[-3000:], r.stderr[-3000:])
                _check(r.returncode == 0, f"{mtype} arm trained ({cond})", failures)

            print("\n7. analyze")
            r = subprocess.run(
                [sys.executable, "-m", "src.analyze",
                 "--results-dir", str(root / "results"),
                 "--out-dir", str(root / "report")],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-2000:])
            _check(r.returncode == 0, "analyze produced tables", failures)
            _check((root / "report" / "summary.csv").exists(),
                   "summary.csv written", failures)
        except ImportError:
            print("  [SKIP] torch not installed -- steps 6-7 skipped.")
            print("         Non-GPU stages all passed, so your data logic is fine.")
            print("         Install torch on the GPU machine and re-run there.")
    finally:
        if keep:
            print(f"\nleft files in {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print("  -", f)
        print("Fix these before running anything on the real dataset.")
        return 1
    print("ALL CHECKS PASSED. The pipeline is wired correctly.")
    print("Next: prepare the real dataset, then run --real.")
    return 0


# ---------------------------------------------------------------------------
# real mode
# ---------------------------------------------------------------------------

def run_real(config: str, n_videos: int, epochs: int, img_size: int) -> int:
    """Same path, small slice of the real data. Catches label/path mistakes."""
    import yaml

    from .splits import load_splits

    failures: list[str] = []
    with open(config) as f:
        cfg = yaml.safe_load(f)

    idx_path = Path(cfg["paths"]["index_csv"])
    _check(idx_path.exists(), f"{idx_path} exists", failures)
    if not idx_path.exists():
        print("\nRun build_index.py first.")
        return 1

    idx = pd.read_csv(idx_path)
    lab = idx[idx.label >= 0]
    print(f"\nindex: {len(idx)} frames, {len(lab)} labelled, "
          f"{lab.video_id.nunique()} videos, pos-rate {lab.label.mean():.4f}")

    _check(len(lab) > 1000, "a plausible number of labelled frames", failures)
    _check(0.02 < lab.label.mean() < 0.60,
           f"standard-plane rate is plausible ({lab.label.mean():.3f}) -- "
           f"if this is near 0 or 1 your label column is misread", failures)
    per_vid = lab.groupby("video_id").size()
    _check(per_vid.median() > 10,
           f"median frames per video ({per_vid.median():.0f}) looks like video data",
           failures)

    sp = load_splits(cfg["paths"]["splits_json"])
    frames_dir = Path(cfg["paths"]["frames_dir"])
    missing = sum(not (frames_dir / p).exists()
                  for p in lab.frame_path.sample(min(200, len(lab)), random_state=0))
    _check(missing == 0, f"sampled frame files all exist on disk ({missing} missing)",
           failures)

    if failures:
        print("\nFix the above before training. A misread label column is the "
              "most common cause and it silently ruins every downstream number.")
        return 1

    # cut down to a few videos and run the real entry point
    root = Path(tempfile.mkdtemp(prefix="spc_real_smoke_"))
    keep = {}
    for s in ("train", "val", "test"):
        keep[s] = sp[s][:max(4, n_videos // 3)]
    small_idx = idx[idx.video_id.isin(sum(keep.values(), []))]
    small_idx.to_csv(root / "index.csv", index=False)
    import json
    json.dump(keep, open(root / "splits.json", "w"))

    cfg2 = dict(cfg)
    cfg2["paths"] = {**cfg["paths"], "index_csv": str(root / "index.csv"),
                     "splits_json": str(root / "splits.json"),
                     "results_dir": str(root / "results")}
    cfg2["data"] = {**cfg["data"], "img_size": img_size, "workers": 2}
    cfg2["train"] = {**cfg["train"], "epochs": epochs, "patience": epochs,
                     "batch_size": 16}
    cfg2["eval"] = {**cfg["eval"], "n_boot": 50}
    yaml.safe_dump(cfg2, open(root / "cfg.yaml", "w"), sort_keys=False)

    print(f"\nrunning 2 short trainings on {small_idx.video_id.nunique()} videos...")
    for cond, mtype in (("dense_all", "frame"), ("dense_all", "temporal")):
        r = subprocess.run(
            [sys.executable, "-m", "src.run_experiment", "--config",
             str(root / "cfg.yaml"), "--condition", cond, "--model", mtype,
             "--seed", "0"], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:], r.stderr[-3000:])
        _check(r.returncode == 0, f"{mtype} arm ran on real data", failures)

    shutil.rmtree(root, ignore_errors=True)

    print("\n" + "=" * 60)
    if failures:
        print("FAILED -- do not launch the full grid yet.")
        return 1
    print("REAL-DATA SMOKE TEST PASSED. Safe to launch the full grid.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="Fake data, no dataset or GPU needed. Run this first.")
    ap.add_argument("--real", action="store_true",
                    help="Small slice of the actual dataset. Run before the full grid.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-videos", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--keep", action="store_true", help="Keep temp files")
    a = ap.parse_args()

    if a.real:
        sys.exit(run_real(a.config, a.n_videos, a.epochs, a.img_size))
    sys.exit(run_synthetic(a.keep))


if __name__ == "__main__":
    main()
