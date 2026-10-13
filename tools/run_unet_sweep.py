"""Run the U-Net variants, so "which one is best" has an answer with a number.

The first segmentation run was a single configuration at a single seed, which
is enough to say the model works and not enough to say anything is better than
anything else. This sweep fills that in along the axes that could plausibly
matter on this corpus:

  seed            0/1/2 at the base config -- the spread that any other
                  difference has to beat to mean anything
  input size      256 vs 384. The pubic symphysis is ~1.5% of a frame, so
                  resolution is the first thing to try for it.
  augmentation    on vs off. 2,575 masks over 266 videos is small.
  loss            CE+Dice vs CE alone. Dice is supposed to be what rescues a
                  1.5%-of-frame class; this measures whether it does.
  pretraining     ImageNet vs random init, to price the shared backbone.
  encoder         ResNet-18 vs ResNet-34.
  input source    the cached 256px frames vs frames re-decoded at the videos'
                  native 512px. The masks ship at 512x512 and so do the
                  source videos, but src/extract_frames.py caches frames at a
                  256px short side -- so the "384px" arm above is feeding a
                  U-Net interpolated pixels, and the native arms are the only
                  ones that can show whether resolution actually helps.

Every arm trains on the same masks, selects on the same 40 validation masks and
is scored on the same 300 test masks, so the comparison is between the one
thing that changed. Runs are sequential: they share one GPU.

    python tools/run_unet_sweep.py                 # all of it, ~2 h
    python tools/run_unet_sweep.py --only base_s0 size384_s0
    python tools/run_unet_sweep.py --dry-run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "unet"

# name -> extra flags on top of the common ones
SWEEP: dict[str, list[str]] = {
    "base_s0":     ["--seed", "0"],
    "base_s1":     ["--seed", "1"],
    "base_s2":     ["--seed", "2"],
    "size384_s0":  ["--seed", "0", "--img-size", "384", "--batch-size", "8"],
    "noaug_s0":    ["--seed", "0", "--no-aug"],
    "ce_only_s0":  ["--seed", "0", "--loss", "ce"],
    "scratch_s0":  ["--seed", "0", "--no-pretrained"],
    "resnet34_s0": ["--seed", "0", "--arch", "resnet34"],
}

# The native arms read data/frames_native (tools/extract_mask_frames.py) and
# skip the derived-classifier stage: that stage scores all 8,665 test frames,
# and only the 2,915 masked ones exist at native resolution.
NATIVE = ["--masks", "data/masks_native.csv",
          "--frames-dir", "data/frames_native", "--skip-classification"]

SWEEP.update({
    "native256_s0": ["--seed", "0", "--img-size", "256"] + NATIVE,
    "native384_s0": ["--seed", "0", "--img-size", "384",
                     "--batch-size", "8"] + NATIVE,
    "native512_s0": ["--seed", "0", "--img-size", "512",
                     "--batch-size", "4"] + NATIVE,
})

COMMON = ["--epochs", "40", "--patience", "8"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-done", action="store_true",
                    help="leave arms that already have a results.json alone")
    a = ap.parse_args()

    names = a.only or list(SWEEP)
    unknown = [n for n in names if n not in SWEEP]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; known: {list(SWEEP)}")

    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    status = {}

    for i, name in enumerate(names, 1):
        run_dir = OUT / name
        if a.skip_done and (run_dir / "results.json").exists():
            print(f"[{i}/{len(names)}] {name}: already done, skipping", flush=True)
            status[name] = "skipped"
            continue

        cmd = [sys.executable, "-m", "src.train_unet",
               "--out-dir", str(run_dir.relative_to(ROOT))] + COMMON + SWEEP[name]
        print(f"\n[{i}/{len(names)}] {name}\n  $ {' '.join(cmd[2:])}", flush=True)
        if a.dry_run:
            status[name] = "dry-run"
            continue

        t0 = time.time()
        p = subprocess.run(cmd, cwd=ROOT)
        dt = (time.time() - t0) / 60
        # One arm failing should not cost the rest of a two-hour sweep.
        status[name] = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
        print(f"[{i}/{len(names)}] {name}: {status[name]} in {dt:.1f} min",
              flush=True)

    print(f"\nsweep finished in {(time.time() - t_all) / 60:.1f} min")
    for k, v in status.items():
        print(f"  {k:<14} {v}")
    return 0 if all(v in ("ok", "skipped", "dry-run")
                    for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
