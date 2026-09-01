"""Run a list of (condition, model, seed) experiments sequentially.

Ordered cheapest-first so the density curve fills in from both ends early and
a crash costs the least finished work. Every run is skipped if its
results.json already exists, so the grid is resumable after an interruption.

    python tools/run_grid.py --plan arm1 --seeds 0
    python tools/run_grid.py --plan arm2 --seeds 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# cheapest first: 434 frames -> 53,996 frames
ARM1 = [
    "sparse_k1", "sparse_k2", "sparse_k5", "sparse_k10", "sparse_k20",
    "dense_matched_k5", "dense_prior_matched_k5",
    "dense_matched_k20", "dense_prior_matched_k20",
    "dense_stride8", "dense_stride4", "dense_stride2", "dense_all",
]
# the headline pair, for the extra seeds
HEADLINE = ["sparse_k1", "sparse_k5", "dense_matched_k5",
            "dense_prior_matched_k5", "dense_all"]
ARM2 = ["dense_all"]

PLANS = {"arm1": ARM1, "headline": HEADLINE, "arm2": ARM2}


def already_done(results_dir: Path, cond: str, model: str, seed: int,
                 tag: str) -> bool:
    name = f"{cond}__{model}{'_' + tag if tag else ''}__seed{seed}"
    return (results_dir / name / "results.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="arm1", choices=sorted(PLANS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--model", default="frame", choices=["frame", "temporal"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default="")
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    conds = PLANS[a.plan]
    results_dir = ROOT / a.results_dir

    jobs = [(c, s) for s in seeds for c in conds]
    print(f"[grid] plan={a.plan} model={a.model} seeds={seeds} "
          f"-> {len(jobs)} runs", flush=True)

    t_start = time.time()
    failures = []
    for i, (cond, seed) in enumerate(jobs, 1):
        if already_done(results_dir, cond, a.model, seed, a.tag):
            print(f"[grid] {i}/{len(jobs)} skip {cond} seed{seed} (done)",
                  flush=True)
            continue
        cmd = [sys.executable, "-m", "src.run_experiment",
               "--config", a.config, "--condition", cond,
               "--model", a.model, "--seed", str(seed)]
        if a.tag:
            cmd += ["--tag", a.tag]
        print(f"[grid] {i}/{len(jobs)} START {cond} {a.model} seed{seed} "
              f"({time.time() - t_start:.0f}s elapsed)", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        dt = time.time() - t0
        if r.returncode != 0:
            failures.append((cond, seed, r.returncode))
            print(f"[grid] FAILED {cond} seed{seed} rc={r.returncode} "
                  f"after {dt:.0f}s", flush=True)
            continue

        name = f"{cond}__{a.model}{'_' + a.tag if a.tag else ''}__seed{seed}"
        try:
            res = json.loads((results_dir / name / "results.json").read_text())
            raw = res["results"]["raw"]
            line = (f"macroF1 {raw['macro_f1']:.4f} balAcc "
                    f"{raw['balanced_accuracy']:.4f} auprc {raw['auprc']:.4f} "
                    f"top1 {raw['top1_frame_precision']:.4f}")
            if "smoothed" in res["results"]:
                line += f" | smoothed {res['results']['smoothed']['macro_f1']:.4f}"
        except Exception as e:  # noqa: BLE001 - reporting only
            line = f"(could not read results.json: {e!r})"
        print(f"[grid] DONE {cond} seed{seed} in {dt:.0f}s -- {line}",
              flush=True)

    print(f"[grid] finished {len(jobs)} runs in "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    if failures:
        print(f"[grid] {len(failures)} FAILURES: {failures}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
