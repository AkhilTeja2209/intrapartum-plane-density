"""Collect results/*/results.json into the compact table the web page renders.

Kept separate from src/analyze.py: that module produces the paper's tables and
significance tests, this one produces a display artefact. Mixing them would
mean a change to the site could quietly alter a reported number.

    python tools/make_site_summary.py --results results --out site/results_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# display order, and the group each condition belongs to
ORDER = [
    ("sparse_k1", "A-sparse"), ("sparse_k2", "A-sparse"),
    ("sparse_k5", "A-sparse"), ("sparse_k10", "A-sparse"),
    ("sparse_k20", "A-sparse"),
    ("dense_stride8", "A-dense"), ("dense_stride4", "A-dense"),
    ("dense_stride2", "A-dense"), ("dense_all", "A-dense"),
    ("dense_matched_k5", "B-budget"), ("dense_matched_k20", "B-budget"),
    ("dense_prior_matched_k5", "B-prior"),
    ("dense_prior_matched_k20", "B-prior"),
]
RANK = {name: i for i, (name, _) in enumerate(ORDER)}
GROUP = dict(ORDER)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="site/results_summary.json")
    ap.add_argument("--model", default="frame")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = []
    for rj in sorted((ROOT / a.results).glob("*/results.json")):
        try:
            r = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        if r.get("model_type") != a.model or int(r.get("seed", -1)) != a.seed:
            continue
        cond = (r.get("condition") or {}).get("name")
        if cond not in RANK:
            continue
        raw = r["results"]["raw"]
        ts = r.get("train_stats", {})
        row = {
            "condition": cond,
            "group": GROUP[cond],
            "train_frames": int(ts.get("n_frames", 0)),
            "train_videos": int(ts.get("n_videos", 0)),
            "pos_rate": round(float(ts.get("pos_rate", 0.0)), 4),
            "macro_f1": round(float(raw["macro_f1"]), 4),
            "balanced_accuracy": round(float(raw["balanced_accuracy"]), 4),
            "auprc": round(float(raw["auprc"]), 4),
            "top1_frame_precision": round(float(raw["top1_frame_precision"]), 4),
        }
        ci = raw.get("macro_f1_ci95")
        if ci:
            row["macro_f1_ci95"] = [round(float(ci[0]), 4), round(float(ci[1]), 4)]
        sm = r["results"].get("smoothed")
        if sm:
            row["macro_f1_smoothed"] = round(float(sm["macro_f1"]), 4)
        rows.append(row)

    rows.sort(key=lambda x: RANK[x["condition"]])
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out} with {len(rows)} conditions")
    for r in rows:
        print(f"  {r['condition']:<26} {r['train_frames']:>7,} frames  "
              f"pos {r['pos_rate']:.3f}  macroF1 {r['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
